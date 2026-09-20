from streamlit.testing.v1 import AppTest
import pytest
import os

def test_app_next_smoke():
    at = AppTest.from_file("../app.py")
    at.run()
    assert not at.exception, f"App crashed with: {at.exception}"

def test_pages_smoke():
    pages = [
        "../app/page_1_command_center.py",
        "../app/page_2_inspect_upload.py",
        "../app/page_3_root_cause.py",
        "../app/page_4_flow_bottleneck.py",
        "../app/page_5_profit_whatif.py",
        "../app/page_6_actions.py",
        "../app/page_7_trust.py",
    ]
    for page in pages:
        at = AppTest.from_file(page)
        at.run()
        assert not at.exception, f"Page {page} crashed with: {at.exception}"
