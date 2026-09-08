import os
print("LOCAL_RANK =", os.environ.get("LOCAL_RANK"), "RANK =", os.environ.get("RANK"))
