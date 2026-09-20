import os

os.makedirs(".streamlit", exist_ok=True)
os.makedirs("app", exist_ok=True)

with open(".streamlit/config.toml", "w") as f:
    f.write("""[theme]
base="dark"
primaryColor="#2DD4BF"
backgroundColor="#0B0F14"
secondaryBackgroundColor="#121923"
textColor="#E6EDF3"
font="sans serif"
""")

with open("app/__init__.py", "w") as f:
    pass
