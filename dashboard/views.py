import os
import re
import json
import csv
import torch
import torch.nn.functional as F
import pandas as pd
from datetime import timedelta
from pythainlp.tokenize import word_tokenize

from django.shortcuts import render, redirect
from django.utils import timezone
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.db.models import Sum
from django.db.models.functions import TruncDate

# LINE Bot SDK
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage,
    TextSendMessage, FlexSendMessage,
    QuickReply, QuickReplyButton, MessageAction
)

# Custom Models & BiLSTM Architecture
from model.model import FinancialMoodBiLSTM
from .models import ExpenseLog, UserBudget

# ----------------------------------------------------
# LINE Credentials
# ----------------------------------------------------
LINE_CHANNEL_SECRET = '5cd4e858aa13f8e6dffc0e30f274962b'
LINE_CHANNEL_ACCESS_TOKEN = 'tO2GhABm2rw4COY1Wh7UU8BNw34SMT+E3utuP9UqnW8KF9H634iGjSL9Q22znDIhv//qJResk1XVpoGPmCddUA0BrYbNhuPxPB+N++CZIaxNx3F87Tw8YiUXd2zk7HgGi0tvSl7kz7JPmwSJ+UhdegdB04t89/1O/w1cDnyilFU='

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# ----------------------------------------------------
# Model Setup
# ----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

with open(os.path.join(MODEL_DIR, 'vocab.json'), 'r', encoding='utf-8') as f:
    vocab = json.load(f)

with open(os.path.join(MODEL_DIR, 'label_encoders.json'), 'r', encoding='utf-8') as f:
    label_encoders = json.load(f)

categories = label_encoders['categories']
moods = label_encoders['moods']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FinancialMoodBiLSTM(
    vocab_size=len(vocab),
    num_categories=len(categories),
    num_moods=len(moods)
)
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'weights.pth'), map_location=device))
model.eval()

SMART_NUDGES = {
    'Stress': "💡 AI Nudge: ช่วงนี้ดูมีความเครียดสะสมนะครับ ลองพักผ่อนหรือออกกำลังกายแทนการใช้เงินแก้เครียดนะ!",
    'Impulse/Peer': "💡 AI Nudge: ช้อปปิ้งตามกระแสหรือเปล่าเอ่ย? ลองกฎ 24 ชั่วโมง พักตะกร้าสินค้าไว้ก่อนสั่งซื้อนะ!",
    'Boredom': "💡 AI Nudge: ซื้อเพราะแค่อยากหาอะไรทำแก้เหงาหรือเปล่า? ลองหากิจกรรมอดิเรกใหม่ๆ ทำดูนะครับ",
    'Happiness/Reward': "🎉 AI Nudge: ให้รางวัลตัวเองได้แต่อย่าลืมแบ่งเก็บออมด้วยนะครับ!",
    'Necessity': "✅ AI Nudge: รายจ่ายจำเป็น บันทึกเรียบร้อยครับ บริหารงบดีๆ นะครับ"
}

# ----------------------------------------------------
# Helper Functions
# ----------------------------------------------------
def tokenize_bilingual(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9ก-๙\s]', '', text)
    if re.search(r'[ก-๙]', text):
        tokens = word_tokenize(text, engine='newmm', keep_whitespace=False)
    else:
        tokens = text.split()
    return tokens

def extract_amount(text):
    match = re.search(r'(\d[\d,]*\.?\d*)\s*(บาท|thb|\$)?', text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1).replace(',', ''))
        except ValueError:
            return 0.0
    return 0.0

def detect_transaction_type(text):
    income_keywords = ['รายรับ', 'เงินเดือน', 'โอนเข้า', 'ได้เงิน', 'รับเงิน', 'โบนัส', 'ขายได้', 'income', 'salary']
    text_lower = text.lower()
    for kw in income_keywords:
        if kw in text_lower:
            return 'Income'
    return 'Expense'

def predict_financial_mood(text):
    amount = extract_amount(text)
    tx_type = detect_transaction_type(text)
    tokens = tokenize_bilingual(text)
    ids = [vocab.get(w, vocab['<UNK>']) for w in tokens]
    if len(ids) < 50:
        ids += [vocab['<PAD>']] * (50 - len(ids))
    else:
        ids = ids[:50]
        
    input_tensor = torch.tensor([ids]).to(device)
    with torch.no_grad():
        cat_preds, mood_preds = model(input_tensor)
        cat_idx = torch.argmax(F.softmax(cat_preds, dim=1), dim=1).item()
        mood_idx = torch.argmax(F.softmax(mood_preds, dim=1), dim=1).item()
        
    return {
        'amount': amount,
        'transaction_type': tx_type,
        'category': 'Income' if tx_type == 'Income' else categories[cat_idx],
        'trigger_mood': 'Income/Investment' if tx_type == 'Income' else moods[mood_idx]
    }

def get_quick_reply_menu():
    return QuickReply(
        items=[
            QuickReplyButton(action=MessageAction(label="📅 สรุปวันนี้", text="สรุปวันนี้")),
            QuickReplyButton(action=MessageAction(label="📆 สรุปสัปดาห์นี้", text="สรุปสัปดาห์นี้")),
            QuickReplyButton(action=MessageAction(label="📊 สรุปเดือนนี้", text="สรุปเดือนนี้")),
            QuickReplyButton(action=MessageAction(label="❓ วิธีใช้งาน", text="วิธีใช้งาน")),
            QuickReplyButton(action=MessageAction(label="⚙️ ตั้งงบ 10000", text="ตั้งงบ 10000")),
        ]
    )

def get_expense_summary(user_id, timeframe="month"):
    now = timezone.now()
    if timeframe == "today":
        start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        title = "📊 สรุปรายจ่ายวันนี้"
    elif timeframe == "week":
        start_date = now - timedelta(days=now.weekday())
        start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        title = "📆 สรุปรายจ่ายสัปดาห์นี้"
    else:
        start_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        title = "📊 สรุปรายจ่ายเดือนนี้"

    logs = ExpenseLog.objects.filter(user_id=user_id, created_at__gte=start_date)
    total_exp = logs.filter(transaction_type='Expense').aggregate(Sum("amount"))["amount__sum"] or 0.0
    total_inc = logs.filter(transaction_type='Income').aggregate(Sum("amount"))["amount__sum"] or 0.0
    count = logs.count()

    return f"{title}\n\n• จำนวนรายการ: {count} รายการ\n• รายรับรวม: {total_inc:,.2f} THB\n• รายจ่ายรวม: {total_exp:,.2f} THB\n• คงเหลือสุทธิ: {(total_inc - total_exp):,.2f} THB"

# ----------------------------------------------------
# Webhook Endpoint
# ----------------------------------------------------
@csrf_exempt
def line_webhook(request):
    if request.method == 'POST':
        signature = request.META.get('HTTP_X_LINE_SIGNATURE', '')
        body = request.body.decode('utf-8')

        try:
            handler.handle(body, signature)
        except InvalidSignatureError:
            return HttpResponse(status=400)

        return HttpResponse("OK", status=200)
    return HttpResponse("Method not allowed", status=405)

# ----------------------------------------------------
# Event: จัดการข้อความ Text
# ----------------------------------------------------
@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    user_id = event.source.user_id
    user_text = event.message.text.strip()

    # 1. คู่มือวิธีใช้งาน
    if any(keyword in user_text for keyword in ["วิธีใช้งาน", "แนะนำ", "คู่มือ", "help"]):
        guide_msg = (
            "📖 คู่มือการใช้งาน Financial & Mood Intelligence Bot\n\n"
            "1️⃣ บันทึกรายการ:\n"
            "• พิมพ์ตามปกติ เช่น 'กินข้าวผัดกระเพรา 60 บาท', 'ช้อปปิ้งเสื้อผ้า 1500 เครียดมาก', 'เงินเดือนออก 25000'\n\n"
            "2️⃣ ดูสรุปรายจ่าย:\n"
            "• กดปุ่ม Quick Reply หรือพิมพ์ 'สรุปวันนี้', 'สรุปสัปดาห์นี้', 'สรุปเดือนนี้'\n\n"
            "3️⃣ ตั้งงบประมาณรายเดือน:\n"
            "• พิมพ์ 'ตั้งงบ' ตามด้วยจำนวนเงิน เช่น 'ตั้งงบ 10000'\n\n"
        )
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=guide_msg, quick_reply=get_quick_reply_menu())
        )
        return

    # 2. ดูสรุปรายช่วงเวลา
    if user_text in ["สรุปวันนี้", "สรุปสัปดาห์นี้", "สรุปเดือนนี้", "สรุป", "report", "summary"]:
        tf = "month"
        if user_text == "สรุปวันนี้":
            tf = "today"
        elif user_text == "สรุปสัปดาห์นี้":
            tf = "week"

        summary_msg = get_expense_summary(user_id, timeframe=tf)
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=summary_msg, quick_reply=get_quick_reply_menu())
        )
        return

    # 3. ตั้งงบประมาณ
    if user_text.startswith("ตั้งงบ"):
        amt = extract_amount(user_text)
        if amt > 0:
            UserBudget.objects.update_or_create(user_id=user_id, defaults={'monthly_budget': amt})
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(
                    text=f"🎯 ตั้งงบประมาณประจำเดือนสำเร็จ: {amt:,.2f} THB",
                    quick_reply=get_quick_reply_menu()
                )
            )
            return

    # 4. ประมวลผลและบันทึกรายการ
    res = predict_financial_mood(user_text)
    if res['amount'] == 0.0:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(
                text="❌ ไม่พบจำนวนเงินในข้อความ กรุณาระบุตัวเลขด้วยครับ เช่น 'กินชาบู 599 บาท'",
                quick_reply=get_quick_reply_menu()
            )
        )
        return

    ExpenseLog.objects.create(
        user_id=user_id,
        text=user_text,
        amount=res['amount'],
        category=res['category'],
        trigger_mood=res['trigger_mood'],
        transaction_type=res['transaction_type']
    )

    # 5. คำนวณ Budget Alert & Color Indicator
    budget_obj = UserBudget.objects.filter(user_id=user_id).first()
    monthly_budget = budget_obj.monthly_budget if budget_obj else 0.0

    now = timezone.now()
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    current_month_total = ExpenseLog.objects.filter(
        user_id=user_id, transaction_type='Expense', created_at__gte=month_start
    ).aggregate(Sum('amount'))['amount__sum'] or 0.0

    card_color = "#2E7D32" if res['transaction_type'] == 'Income' else "#1DB446"
    budget_warning = ""

    if res['transaction_type'] == 'Expense' and monthly_budget > 0:
        usage_pct = (current_month_total / monthly_budget) * 100
        if usage_pct >= 100:
            card_color = "#D32F2F"
            budget_warning = f"⚠️ ใช้เงินเกินงบแล้ว! ({current_month_total:,.0f}/{monthly_budget:,.0f} THB)"
        elif usage_pct >= 80:
            card_color = "#F57C00"
            budget_warning = f"⚡ ใช้เงินไปแล้ว {usage_pct:.0f}% ของงบเดือนนี้"

    # 6. Emotional Nudge
    recent_moods = list(
        ExpenseLog.objects.filter(user_id=user_id)
        .order_by("-created_at")[:3]
        .values_list("trigger_mood", flat=True)
    )
    stress_count = sum(1 for m in recent_moods if m and ("เครียด" in m or "Stress" in m))
    
    nudge_msg = SMART_NUDGES.get(res['trigger_mood'], "")
    if stress_count >= 3:
        nudge_msg = "💙 ช่วงนี้เห็นคุณใช้จ่ายตอนเครียดบ่อย มีเรื่องกังวลรึเปล่า? ลองพักผ่อนสักหน่อยนะ"
    elif budget_warning:
        nudge_msg = budget_warning

    title_text = "💰 บันทึกรายรับสำเร็จ!" if res['transaction_type'] == 'Income' else "💸 บันทึกรายจ่ายสำเร็จ!"
    amount_val = res.get('amount') or 0.0
    category_val = str(res.get('category') or "ทั่วไป")
    mood_val = str(res.get('trigger_mood') or "ปกติ")
    nudge_val = str(nudge_msg or "").strip()

    body_contents = [
        {"type": "text", "text": title_text, "weight": "bold", "size": "lg", "color": card_color},
        {"type": "text", "text": f"\"{user_text}\"", "size": "xs", "color": "#888888", "wrap": True, "margin": "sm"},
        {"type": "separator", "margin": "lg"},
        {
            "type": "box", "layout": "vertical", "margin": "lg", "spacing": "sm",
            "contents": [
                {
                    "type": "box", "layout": "baseline",
                    "contents": [
                        {"type": "text", "text": "จำนวนเงิน", "color": "#aaaaaa", "size": "sm", "flex": 2},
                        {"type": "text", "text": f"{amount_val:,.2f} THB", "weight": "bold", "color": "#333333", "size": "sm", "flex": 3, "align": "end"}
                    ]
                },
                {
                    "type": "box", "layout": "baseline",
                    "contents": [
                        {"type": "text", "text": "หมวดหมู่", "color": "#aaaaaa", "size": "sm", "flex": 2},
                        {"type": "text", "text": category_val, "weight": "bold", "color": "#0288D1", "size": "sm", "flex": 3, "align": "end"}
                    ]
                },
                {
                    "type": "box", "layout": "baseline",
                    "contents": [
                        {"type": "text", "text": "อารมณ์/สาเหตุ", "color": "#aaaaaa", "size": "sm", "flex": 2},
                        {"type": "text", "text": mood_val, "weight": "bold", "color": "#F57C00", "size": "sm", "flex": 3, "align": "end"}
                    ]
                }
            ]
        },
        {"type": "separator", "margin": "lg"},
    ]

    if nudge_val:
        body_contents.append({"type": "text", "text": nudge_val, "size": "xs", "color": "#E65100", "wrap": True, "margin": "md"})

    flex_content = {
        "type": "bubble",
        "body": {
            "type": "box", "layout": "vertical",
            "contents": body_contents
        }
    }

    line_bot_api.reply_message(
        event.reply_token,
        FlexSendMessage(
            alt_text="ผลการบันทึกรายการ",
            contents=flex_content,
            quick_reply=get_quick_reply_menu()
        )
    )

# ----------------------------------------------------
# Web Dashboard View & Export CSV
# ----------------------------------------------------
def index(request):
    logs = ExpenseLog.objects.all().order_by('-created_at')
    total_exp = logs.filter(transaction_type='Expense').aggregate(Sum('amount'))['amount__sum'] or 0.0
    total_inc = logs.filter(transaction_type='Income').aggregate(Sum('amount'))['amount__sum'] or 0.0

    cat_summary = list(logs.filter(transaction_type='Expense').values('category').annotate(total=Sum('amount')))
    mood_summary = list(logs.filter(transaction_type='Expense').values('trigger_mood').annotate(total=Sum('amount')))

    seven_days_ago = timezone.now() - timedelta(days=7)
    daily_stats = (
        ExpenseLog.objects.filter(transaction_type='Expense', created_at__gte=seven_days_ago)
        .annotate(date=TruncDate('created_at'))
        .values('date')
        .annotate(daily_total=Sum('amount'))
        .order_by('date')
    )

    dates = [item['date'].strftime('%d/%m') for item in daily_stats]
    totals = [float(item['daily_total']) for item in daily_stats]

    context = {
        'expenses': logs,
        'recent_logs': logs[:15],
        'total_exp': total_exp,
        'total_inc': total_inc,
        'net_balance': total_inc - total_exp,
        'cat_summary_json': json.dumps(cat_summary),
        'mood_summary_json': json.dumps(mood_summary),
        'chart_dates': json.dumps(dates),
        'chart_totals': json.dumps(totals),
    }
    return render(request, 'index.html', context)

def export_csv(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="financial_mood_report.csv"'
    
    # UTF-8 BOM สำหรับ Excel ภาษาไทย
    response.write(u'\ufeff'.encode('utf8'))

    writer = csv.writer(response)
    writer.writerow(['ID', 'Timestamp', 'User ID', 'Type', 'Text', 'Amount', 'Category', 'Mood'])

    expenses = ExpenseLog.objects.all().order_by('-created_at')
    for item in expenses:
        writer.writerow([
            item.id,
            item.created_at.strftime('%Y-%m-%d %H:%M:%S') if hasattr(item, 'created_at') else '',
            getattr(item, 'user_id', ''),
            getattr(item, 'transaction_type', ''),
            getattr(item, 'text', ''),
            getattr(item, 'amount', 0),
            getattr(item, 'category', ''),
            getattr(item, 'trigger_mood', '')
        ])

    return response

def clear_data(request):
    if request.method == 'POST':
        ExpenseLog.objects.all().delete()
    return redirect('index')