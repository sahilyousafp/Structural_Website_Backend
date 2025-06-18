import os

# Path to the Rhino 3dm file; configurable via environment variable
RHINO_PATH = os.getenv('RHINO_PATH', r".\basicformMETERS.3dm")
SUPABASE_URL = os.getenv('SUPABASE_URL', 'https://apdbfbjnlsxjfubqahtl.supabase.co')
SUPABASE_KEY = os.getenv('SUPABASE_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFwZGJmYmpubHN4amZ1YnFhaHRsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDc0NzY1MzgsImV4cCI6MjA2MzA1MjUzOH0.DPINyYHHUzcuQ6AOcp8hh1W1eIiamOFPKFRMNfHypSU')
SUPABASE_BUCKET = os.getenv('SUPABASE_BUCKET', 'models')
