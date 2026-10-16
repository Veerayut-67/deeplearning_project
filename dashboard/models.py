from django.db import models

class ExpenseLog(models.Model):
    TRANSACTION_TYPES = [
        ('Expense', 'รายจ่าย'),
        ('Income', 'รายรับ'),
    ]

    user_id = models.CharField(max_length=100, db_index=True)
    text = models.TextField()
    amount = models.FloatField()
    category = models.CharField(max_length=50)
    trigger_mood = models.CharField(max_length=50)
    transaction_type = models.CharField(max_length=10, choices=TRANSACTION_TYPES, default='Expense')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"[{self.transaction_type}] {self.user_id} | {self.amount} THB | {self.category} ({self.trigger_mood})"

class UserBudget(models.Model):
    user_id = models.CharField(max_length=100, unique=True)
    monthly_budget = models.FloatField(default=10000.0)  # งบประจำเดือน (ค่าเริ่มต้น 10,000 บาท)

    def __str__(self):
        return f"{self.user_id} | Budget: {self.monthly_budget} THB"