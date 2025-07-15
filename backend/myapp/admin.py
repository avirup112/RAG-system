from django.contrib import admin
from .models import *

# Register your models here.
admin.site.register(OCRDocument)
admin.site.register(UploadedDocument)
admin.site.register(DocumentChunk)
admin.site.register(Clause)
admin.site.regsiter(ChunkEmbedding)
admin.site.register(LegalDocument)

