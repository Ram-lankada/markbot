import os , time
from dotenv import load_dotenv

load_dotenv() 

_script_dir = os.path.dirname(os.path.abspath(__file__))
_default_export_dir = os.path.abspath(os.path.join(os.path.dirname(_script_dir), "exports"))
EXPORT_DIR = os.path.abspath(os.environ.get("EXPORT_DIR", _default_export_dir))
OUT_MD = os.environ.get("OUT_MD", "./out/mindmap.md")
OUT_DIR = (os.environ.get("OUT_DIR"))

# OUT_DIR is not converted to an absolute path, and may be None if not set in environment.
# This will cause os.path.join(OUT_DIR, ...) to fail with TypeError if OUT_DIR is None.
# Additionally, OUT_DIR should likely be an absolute path (as in other configs).
# Correct version:
if not OUT_DIR:
    raise ValueError("OUT_DIR environment variable must be set")
out_path = os.path.join(os.path.abspath(OUT_DIR), "mindmap.md")
print(out_path)
print(OUT_DIR) 
print(EXPORT_DIR) 
