from django.urls import path
from .views import *

urlpatterns = [
    path('ask/', ask_legal_bot),
    path("summarize/", summarize_document, name="summarize_document"),
    path('extract_clauses/', extract_clauses, name='extract_clauses'),
    path('filter_clauses/', filter_clauses, name='filter_clauses'),
]
