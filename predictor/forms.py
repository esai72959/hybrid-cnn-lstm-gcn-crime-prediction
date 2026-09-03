import re

from django import forms
from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError

User = get_user_model()

MOBILE_RE = re.compile(r"^[0-9]{10}$")


class LoginForm(forms.Form):
    email = forms.EmailField()
    password = forms.CharField(widget=forms.PasswordInput)
    remember = forms.BooleanField(required=False)

    # `non_field_errors` (bad credentials, inactive account, etc.) is what
    # login.html renders in the `.auth-alert--error` block.


class SignupForm(forms.Form):
    full_name = forms.CharField(min_length=2, max_length=150)
    email = forms.EmailField()
    mobile = forms.CharField()
    password = forms.CharField(widget=forms.PasswordInput)
    confirm_password = forms.CharField(widget=forms.PasswordInput)
    terms = forms.BooleanField(required=True, error_messages={
        "required": "You must accept the Terms & Conditions to continue."
    })

    def clean_email(self):
        email = self.cleaned_data["email"].strip().lower()
        if User.objects.filter(email__iexact=email).exists():
            raise ValidationError("An account with this email already exists.")
        return email

    def clean_mobile(self):
        mobile = self.cleaned_data["mobile"].strip()
        if not MOBILE_RE.match(mobile):
            raise ValidationError("Enter a valid 10-digit mobile number.")
        return mobile

    def clean_password(self):
        password = self.cleaned_data["password"]
        if len(password) < 8:
            raise ValidationError("Password must be at least 8 characters.")
        if not re.search(r"[A-Z]", password):
            raise ValidationError("Password must contain an uppercase letter.")
        if not re.search(r"[0-9]", password):
            raise ValidationError("Password must contain a number.")
        if not re.search(r"[^A-Za-z0-9]", password):
            raise ValidationError("Password must contain a special character.")
        return password

    def clean(self):
        cleaned = super().clean()
        password = cleaned.get("password")
        confirm = cleaned.get("confirm_password")
        if password and confirm and password != confirm:
            self.add_error("confirm_password", "Passwords do not match.")
        return cleaned
