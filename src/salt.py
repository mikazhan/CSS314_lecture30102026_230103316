import hashlib
import os

def get_student_params(student_id):
    h = int(hashlib.sha256(student_id.encode("utf-8")).hexdigest()[:8], 16)
    N = 1024 + ((h % 9) * 64)
    buffer_elements = 16777216
    seed = h % 100007

    return {
        "student_id": student_id,
        "N": N,
        "buffer_elements": buffer_elements,
        "seed": seed
    }