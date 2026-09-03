"""
Email-based authentication backend.

Your login form only collects `email` + `password`, but Django's default
User model authenticates by `username`. This backend lets users log in
with their email while still using the stock User model (we store the
email in the `username` field too — see forms.py).

Add to settings.py:

    AUTHENTICATION_BACKENDS = [
        "yourapp.backends.EmailBackend",
        "django.contrib.auth.backends.ModelBackend",  # keep as fallback
    ]
"""

from django.contrib.auth import get_user_model
from django.contrib.auth.backends import ModelBackend

User = get_user_model()


class EmailBackend(ModelBackend):
    def authenticate(self, request, username=None, password=None, **kwargs):
        email = kwargs.get("email", username)
        if email is None or password is None:
            return None
        try:
            user = User.objects.get(email__iexact=email)
        except User.DoesNotExist:
            User().set_password(password)  # mitigate timing attack
            return None
        except User.MultipleObjectsReturned:
            user = User.objects.filter(email__iexact=email).order_by("id").first()

        if user.check_password(password) and self.user_can_authenticate(user):
            return user
        return None
