from django.db import models
import pickle
from django.utils import timezone
# Create your models here.

class UploadedDocument(models.Model):
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to='documents/')
    full_text = models.TextField()
    index_path = models.CharField(max_length=500)
    chunks_path = models.CharField(max_length=500)
    created_at = models.DateTimeField(default=timezone.now)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    def __str__ (self):
        return self.title
    
class DocumentChunk(models.Model):
    document = models.ForeignKey(UploadedDocument, related_name='chunks', on_delete=models.CASCADE)
    chunk_text = models.TextField()
    chunk_index = models.PositiveIntegerField()
    
    def __str__(self):
        return f"Chunk {self.document.title} - Chunk{self.chunk_index}"
    
class Clause(models.Model):
    document = models.ForeignKey(UploadedDocument, related_name='clauses', on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    content = models.TextField()
    start_index = models.IntegerField(null=True, blank=True)
    end_index = models.IntegerField(null=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now)
    
class ChunkEmbedding(models.Model):
    chunk = models.OneToOneField(DocumentChunk, related_name='embedding', on_delete=models.CASCADE)
    embedding_vector = models.BinaryField()
    
    def __str__(self):
        return f"Embedding for {self.chunk}"
    
class LegalDocument(models.Model):
    title = models.CharField(max_length=255)
    content = models.TextField()
    document_type = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    user_name = models.CharField(max_length=255, blank=True)
    download_format = models.CharField(max_length=10, default="txt")  # txt, pdf, docx

    def __str__(self):
        return self.title
    

class OCRDocument(models.Model):
    filename = models.Charfield(max_length=255)
    extracted_text = models.TextField()
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.filename