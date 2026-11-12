from flask import Flask, request, redirect, url_for, session, render_template_string, send_file
from html.parser import HTMLParser
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from werkzeug.security import generate_password_hash, check_password_hash
from database.db import get_connection
import html
import io
import json
import os
import re
import zipfile
import requests
import pandas as pd
from urllib.parse import urlparse
import subprocess
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
app = Flask(__name__)
app.secret_key = "intelligrade-secret-key"
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
BULK_REPORT_CACHE = {}
# DATABASE QUERY FUNCTION
def execute_query(query, values=None, fetch=False):
    connection = get_connection()
    if connection is None:
        return None
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute(query, values or ())
        if fetch:
            result = cursor.fetchall()
        else:
            connection.commit()
            result = True
        return result
    except Exception as error:
        connection.rollback()
        print("Database Error:", error)
        return None
    finally:
        cursor.close()
        connection.close()
def execute_insert(query, values=None):
    """Execute an INSERT and return its generated primary-key value."""
    connection = get_connection()
    if connection is None:
        return None
    cursor = connection.cursor()
    try:
        cursor.execute(query, values or ())
        connection.commit()
        return cursor.lastrowid
    except Exception as error:
        connection.rollback()
        print("Database Insert Error:", error)
        return None
    finally:
        cursor.close()
        connection.close()
def assignment_deadline_passed(deadline):
    """Return True when an assignment deadline has passed; blank deadlines stay open."""
    if not deadline:
        return False
    if isinstance(deadline, datetime):
        deadline_value = deadline
    else:
        deadline_text = str(deadline).strip()
        if not deadline_text:
            return False
        try:
            deadline_value = datetime.fromisoformat(deadline_text.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return False
    # Database datetimes are normally local/naive. If a timezone is present,
    # compare using the current local timezone rather than raising an error.
    now = datetime.now(deadline_value.tzinfo) if deadline_value.tzinfo else datetime.now()
    return now > deadline_value


def fetch_github_python_code(repository_url, requested_file=""):
    """Fetch one Python source file from a public GitHub repository."""
    parsed = urlparse(str(repository_url).strip())
    if parsed.scheme not in ("https", "http") or parsed.netloc.lower() not in ("github.com", "www.github.com"):
        raise ValueError("Enter a valid public GitHub repository URL.")
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) != 2:
        raise ValueError("Use a repository URL in the format https://github.com/owner/repository.")
    owner, repository = parts
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", owner) or not re.fullmatch(r"[A-Za-z0-9_.-]+", repository):
        raise ValueError("Invalid GitHub repository URL.")
    api_url = f"https://api.github.com/repos/{owner}/{repository}"
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "IntelliGrade"}
    response = requests.get(api_url, headers=headers, timeout=(5, 15))
    if response.status_code == 404:
        raise ValueError("GitHub repository was not found or is private.")
    response.raise_for_status()
    repository_data = response.json()
    branch = repository_data.get("default_branch")
    if not branch:
        raise ValueError("Could not identify the repository's default branch.")
    archive_url = f"https://api.github.com/repos/{owner}/{repository}/zipball/{branch}"
    archive_response = requests.get(archive_url, headers=headers, timeout=(5, 25))
    archive_response.raise_for_status()
    if len(archive_response.content) > 20 * 1024 * 1024:
        raise ValueError("Repository archive is larger than the 20 MB limit.")
    with zipfile.ZipFile(io.BytesIO(archive_response.content)) as archive:
        python_files = []
        for name in archive.namelist():
            normalized = name.replace("\\", "/")
            if normalized.endswith("/") or "/." in normalized:
                continue
            if normalized.lower().endswith(".py"):
                relative = normalized.split("/", 1)[1] if "/" in normalized else normalized
                python_files.append((relative, name))
        if requested_file:
            wanted = str(requested_file).strip().replace("\\", "/").lstrip("/")
            matches = [item for item in python_files if item[0] == wanted]
            if len(matches) != 1:
                raise ValueError("The specified Python file was not found in the repository.")
            selected = matches[0]
        else:
            root_files = [item for item in python_files if "/" not in item[0]]
            preferred = [item for item in root_files if item[0].lower() in ("main.py", "solution.py")]
            if len(preferred) == 1:
                selected = preferred[0]
            elif len(root_files) == 1:
                selected = root_files[0]
            elif len(python_files) == 1:
                selected = python_files[0]
            else:
                raise ValueError("Multiple Python files found. Add the exact relative file path in the Excel file_path column.")
        if archive.getinfo(selected[1]).file_size > 512 * 1024:
            raise ValueError("The selected Python source file exceeds the 512 KB limit.")
        source = archive.read(selected[1]).decode("utf-8-sig")
        if not source.strip():
            raise ValueError("The selected Python file is empty.")
        if len(source.encode("utf-8")) > 512 * 1024:
            raise ValueError("The selected Python source file exceeds the 512 KB limit.")
        return source, selected[0]
def evaluate_descriptive_answer(student_answer, reference_answer, max_marks):
    """Estimate descriptive-answer marks using TF-IDF cosine similarity."""
    student_answer = str(student_answer or "").strip()
    reference_answer = str(reference_answer or "").strip()
    if not student_answer or not reference_answer:
        return 0.0, 0.0
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        vectors = vectorizer.fit_transform([student_answer, reference_answer])
        similarity = float(cosine_similarity(vectors[0:1], vectors[1:2])[0][0])
        similarity = max(0.0, min(similarity, 1.0))
        awarded_marks = round(similarity * float(max_marks), 2)
        return awarded_marks, round(similarity * 100, 2)
    except ValueError:
        # This can happen when both answers contain no usable vocabulary.
        return 0.0, 0.0
# COMMON HTML AND CSS
def run_python_test_case(source_code, test_input, timeout_seconds=3):
    """Run untrusted Python code in a restricted Docker container.
    Requires Docker and the python:3.12-slim image to be available.
    """
    docker = os.environ.get("DOCKER_EXECUTABLE", "docker")
    with tempfile.TemporaryDirectory(prefix="intelligrade_") as temp_dir:
        work_dir = Path(temp_dir)
        code_file = work_dir / "main.py"
        code_file.write_text(source_code, encoding="utf-8")
        command = [
            docker, "run", "--rm", "-i",
            "--network", "none",
            "--memory", "128m",
            "--cpus", "0.5",
            "--pids-limit", "32",
            "--read-only",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--user", "65534:65534",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
            "--mount", f"type=bind,source={code_file},target=/app/main.py,readonly",
            "--workdir", "/app",
            "python:3.12-slim", "python", "-I", "/app/main.py"
        ]
        try:
            completed = subprocess.run(
                command, input=test_input, text=True,
                capture_output=True, timeout=timeout_seconds,
                check=False
            )
            return {
                "success": completed.returncode == 0,
                "output": completed.stdout.strip(),
                "error": completed.stderr.strip()[:1000],
                "timed_out": False
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "output": "", "error": "Time limit exceeded.", "timed_out": True}
        except FileNotFoundError:
            return {"success": False, "output": "", "error": "Docker was not found. Install and start Docker Desktop.", "timed_out": False}
        except Exception as error:
            return {"success": False, "output": "", "error": str(error)[:1000], "timed_out": False}
PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>IntelliGrade</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {
            box-sizing: border-box;
        }
        body {
            margin: 0;
            min-height: 100vh;
            font-family: Arial, sans-serif;
            background: linear-gradient(
                120deg,
                #392b8b,
                #673477,
                #a63e55
            );
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 25px;
        }
        .container {
            background: white;
            width: 100%;
            max-width: 650px;
            padding: 35px;
            border-radius: 12px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.20);
        }
        .container:has(.dashboard-cards) {
            max-width: 1120px;
            padding: 48px 54px;
        }
        .container:has(.assignment-grid) {
            width: 100%;
            max-width: 1600px;
            padding: 52px 20px;
        }
        @media (max-width: 900px) {
            .container:has(.dashboard-cards),
            .container:has(.assignment-grid) {
                padding: 30px 24px;
            }
        }
        .container:has(.bulk-github-page) {
            max-width: 1120px;
            padding: 42px 48px;
        }
        .bulk-github-page { width: 100%; }
        .bulk-report-actions { display:flex; justify-content:center; gap:12px; flex-wrap:wrap; margin:18px 0 24px; }
        .bulk-download-button, .bulk-secondary-button { display:inline-block; padding:11px 18px; border-radius:6px; text-decoration:none; font-weight:600; }
        .bulk-download-button { background:#2878e8; color:#fff !important; }
        .bulk-secondary-button { background:#eef3fb; color:#2469c8 !important; }
        .bulk-github-page h2 { margin: 0 0 18px; text-align: center; }
        .bulk-github-page form { max-width: 760px; margin: 22px auto 28px; }
        .bulk-github-page .message { text-align: center; margin: 14px 0; }
        .bulk-report { width: 100%; margin: 28px 0 22px; }
        .bulk-report h3 { margin: 0 0 14px; }
        .bulk-report-table-wrap { width: 100%; overflow-x: auto; border: 1px solid #dfe3eb; border-radius: 10px; }
        .bulk-report-table { width: 100%; min-width: 760px; border-collapse: collapse; table-layout: fixed; }
        .bulk-report-table th { background: #f2f5fb; color: #263247; font-weight: 700; text-align: left; }
        .bulk-report-table th, .bulk-report-table td { padding: 14px 16px; border: 1px solid #cbd5e1; vertical-align: top; line-height: 1.5; overflow-wrap: anywhere; }
        .bulk-report-table tbody tr:last-child td { border-bottom: 0; }
        .bulk-report-table tbody tr:nth-child(even) { background: #fafbfe; }
        .bulk-detail { margin-top: 10px; }
        .bulk-detail summary { color: #2563eb; cursor: pointer; font-weight: 600; }
        .bulk-detail pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #f3f4f6; color: #111827; padding: 10px; border-radius: 6px; max-height: 280px; overflow: auto; font-size: 12px; }
        .bulk-detail-table-wrap { overflow-x: auto; margin-top: 8px; }
        .bulk-detail-table { width: 100%; border-collapse: collapse; font-size: 12px; }
        .bulk-detail-table th, .bulk-detail-table td { border: 1px solid #cbd5e1; padding: 7px; vertical-align: top; }
        .bulk-detail-table th { background: #eef3fb; }
        .bulk-detail-table pre { white-space: pre-wrap; overflow-wrap: anywhere; margin: 0; font-size: 11px; }
        .bulk-note { margin: 6px 0; color: #7c4a03; font-size: 12px; }
        .bulk-report-table th:nth-child(1), .bulk-report-table td:nth-child(1) { width: 19%; }
        .bulk-report-table th:nth-child(2), .bulk-report-table td:nth-child(2) { width: 19%; }
        .bulk-report-table th:nth-child(3), .bulk-report-table td:nth-child(3) { width: 15%; white-space: nowrap; }
        .bulk-report-table th:nth-child(4), .bulk-report-table td:nth-child(4) { width: 47%; }
        .container:has(.bulk-github-page) .back-link { display: block; text-align: center; margin-top: 22px; }
        @media (max-width: 900px) {
            .container:has(.bulk-github-page) { padding: 28px 20px; }
            .bulk-report-table { min-width: 680px; }
        }
        .container:has(.my-submissions-page) {
            max-width: 1180px;
            padding: 42px 48px;
        }
        .my-submissions-page { width: 100%; }
        .submission-filters { display: flex; flex-wrap: wrap; justify-content: center; gap: 12px; margin: 0 0 24px; }
        .submission-filter { display: inline-block; padding: 10px 20px; border: 1px solid #cbd5e1; border-radius: 7px; background: #f7f9fc; color: #263247; font-weight: 600; }
        .submission-filter:hover, .submission-filter.active { background: #347fe5; border-color: #347fe5; color: white; text-decoration: none; }
        .my-submissions-page h2 { text-align: center; margin: 0 0 24px; }
        .my-submissions-table-wrap { width: 100%; overflow-x: auto; border: 1px solid #cbd5e1; border-radius: 10px; }
        .my-submissions-table { width: 100%; min-width: 850px; border-collapse: collapse; table-layout: fixed; }
        .my-submissions-table th, .my-submissions-table td { padding: 14px 16px; border: 1px solid #cbd5e1; text-align: left; vertical-align: middle; line-height: 1.45; overflow-wrap: anywhere; }
        .my-submissions-table th { background: #eef3fb; color: #263247; font-weight: 700; white-space: nowrap; }
        .my-submissions-table tbody tr:nth-child(even) td { background: #f8faff; }
        .my-submissions-table tbody tr:hover td { background: #f0f6ff; }
        .my-submissions-table th:nth-child(1), .my-submissions-table td:nth-child(1) { width: 28%; }
        .my-submissions-table th:nth-child(2), .my-submissions-table td:nth-child(2) { width: 14%; }
        .my-submissions-table th:nth-child(3), .my-submissions-table td:nth-child(3) { width: 14%; white-space: nowrap; }
        .my-submissions-table th:nth-child(4), .my-submissions-table td:nth-child(4) { width: 15%; }
        .my-submissions-table th:nth-child(5), .my-submissions-table td:nth-child(5) { width: 19%; white-space: nowrap; }
        .my-submissions-table th:nth-child(6), .my-submissions-table td:nth-child(6) { width: 10%; }
        .my-submissions-page .back-link { display: block; text-align: center; margin-top: 24px; }
        @media (max-width: 900px) {
            .container:has(.my-submissions-page) { padding: 28px 20px; }
            .my-submissions-table { min-width: 780px; }
        }
        .container:has(.history-page) {
            max-width: 1200px;
            padding: 42px;
        }
        .history-page { width: 100%; }
        .history-page h2 { margin: 0 0 18px; }
        .history-page h3 { margin: 0 0 12px; font-size: 20px; }
        .history-page .history-note { margin: 0 0 22px; color: #555; }
        .history-table-wrap { width: 100%; overflow-x: auto; border: 1px solid #e2e8f0; border-radius: 9px; }
        .history-table { width: 100%; min-width: 900px; border-collapse: collapse; table-layout: auto; }
        .history-table th { background: #eef4ff; color: #243b61; font-weight: 700; white-space: nowrap; }
        .history-table th, .history-table td { padding: 14px 16px; text-align: left; border-bottom: 1px solid #d6deea; border-right: 1px solid #d6deea; vertical-align: middle; font-size: 14px; }
        .history-table th:last-child, .history-table td:last-child { border-right: 0; }
        .history-table thead th { border-bottom: 1px solid #c8d4e5; }
        .history-table tbody td { background: #fff; }
        .history-table tbody tr:nth-child(even) td { background: #fafcff; }
        .history-table td { line-height: 1.5; }
        .history-table tr:last-child td { border-bottom: 0; }
        .history-table tbody tr:hover { background: #f8fbff; }
        .history-table .marks-cell, .history-table .date-cell { white-space: nowrap; }
        .history-table .student-cell { min-width: 150px; }
        .history-table .type-cell { min-width: 130px; }
        .history-table .details-cell { text-align: center; }
        .history-table .details-link { display: inline-block; padding: 7px 15px; border-radius: 6px; background: #eaf2ff; font-weight: 600; }
        .history-page .back-link { display: inline-block; margin-top: 24px; padding: 10px 18px; border-radius: 6px; background: #347fe5; color: white; }
        .history-page .back-link:hover { background: #2465c2; text-decoration: none; }
        @media (max-width: 700px) {
            body { padding: 12px; align-items: flex-start; }
            .container:has(.history-page) { padding: 20px 14px; }
            .history-table { min-width: 850px; }
        }
        .container:has(.submission-page) { max-width: 900px; padding: 38px 42px; }
        .submission-page { width: 100%; }
        .submission-page h2 { margin: 0 0 24px; }
        .submission-meta { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 12px 18px; margin: 0 0 24px; }
        .submission-meta p { margin: 0; padding: 11px 13px; background: #f7f9fc; border: 1px solid #dce3ee; border-radius: 7px; }
        .submission-section { margin-top: 22px; }
        .submission-section h3 { margin: 0 0 12px; color: #26364d; }
        .answer-item { border: 1px solid #d6deea; border-radius: 8px; padding: 14px 16px; margin: 10px 0; background: #fff; }
        .answer-item p { margin: 8px 0 0; white-space: pre-wrap; overflow-wrap: anywhere; }
        .submitted-code { white-space: pre-wrap; overflow-wrap: anywhere; background: #f4f5f7; border: 1px solid #d6deea; border-radius: 8px; padding: 16px; line-height: 1.55; overflow-x: auto; }
        .submission-page .back-link { display: inline-block; margin-top: 22px; padding: 10px 18px; border-radius: 6px; background: #347fe5; color: white; }
        @media (max-width: 700px) { .container:has(.submission-page) { padding: 16px 16px; } .submission-meta { grid-template-columns: 1fr; } }
        h1 {
            text-align: center;
            color: #2878df;
            margin-bottom: 25px;
        }
        h2 {
            color: #2878df;
            text-align: center;
        }
        h3 {
            color: #333;
        }
        p {
            color: #444;
            line-height: 1.6;
        }
        select[name="role"]:invalid {
            color: #757575;
        }
        input, select, textarea {
            width: 100%;
            padding: 12px;
            margin: 9px 0;
            border: 1px solid #ccc;
            border-radius: 5px;
            font-size: 14px;
            outline: none;
        }
        input:focus, select:focus, textarea:focus {
            border-color: #347fe5;
            box-shadow: 0 0 4px rgba(52, 127, 229, 0.3);
        }
        textarea {
            min-height: 100px;
            resize: vertical;
        }
        button {
            width: 100%;
            padding: 12px;
            margin-top: 12px;
            background: #347fe5;
            color: white;
            border: none;
            border-radius: 5px;
            font-size: 15px;
            cursor: pointer;
        }
        button:hover {
            background: #2465c2;
        }
        a {
            color: #2878df;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        .message {
            color: #dc2626;
            text-align: center;
        }
        .success {
            color: #15803d;
            text-align: center;
        }
        .welcome {
            text-align: center;
            margin-bottom: 25px;
        }
        .home-actions {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 20px;
            margin-top: 25px;
        }
        .home-btn {
            display: inline-block;
            width: 160px;
            padding: 13px 20px;
            background: #347fe5;
            color: white;
            text-align: center;
            border-radius: 6px;
            font-size: 16px;
        }
        .home-btn:hover {
            background: #2465c2;
            color: white;
            text-decoration: none;
        }
        .page-link {
            display: block;
            margin-top: 18px;
            text-align: center;
        }
        .dashboard-cards {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 24px;
            margin: 30px 0;
        }
        .dashboard-card {
            display: flex;
            flex-direction: column;
            justify-content: center;
            background: #f5f7fb;
            border: 1px solid #e2e7f0;
            border-radius: 12px;
            padding: 30px 22px;
            text-align: center;
            color: #202b45;
            min-height: 175px;
            transition: transform .15s ease, box-shadow .15s ease;
        }
        .dashboard-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 5px 16px rgba(35, 45, 75, .12);
            text-decoration: none;
            color: #245fbd;
        }
        .dashboard-card strong {
            display: block;
            color: #286ed0;
            font-size: 17px;
            margin-bottom: 9px;
        }
        .dashboard-card span {
            display: block;
            color: #606b7c;
            font-size: 13px;
        }
        /* Center the final card when a dashboard has an odd number of cards. */
        .dashboard-cards > .dashboard-card:last-child:nth-child(odd) {
            grid-column: 1 / -1;
            justify-self: center;
            width: calc((100% - 24px) / 2);
        }
        .assignment-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 16px;
            margin-top: 26px;
            align-items: stretch;
        }
        .assignment-card {
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            background: #f5f7fb;
            padding: 25px 20px;
            margin-bottom: 0;
            border: 1px solid #e2e7f0;
            border-radius: 12px;
            min-width: 0;
            min-height: 270px;
            gap: 12px;
        }
        .assignment-card p {
            margin: 4px 0;
            line-height: 1.5;
            min-height: 24px;
        }
        .assignment-card .page-link {
            margin-top: auto;
            padding-top: 8px;
        }
        .assignment-actions {
            display: flex;
            gap: 8px;
            width: 100%;
            margin-top: 8px;
        }
        .assignment-actions a {
            flex: 1;
            min-width: 0;
            text-align: center;
            padding: 11px 5px;
            border-radius: 7px;
            line-height: 1.3;
            font-size: 13px;
            overflow-wrap: normal;
            word-break: normal;
        }
        @media (max-width: 900px) {
            .dashboard-cards, .assignment-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        @media (max-width: 600px) {
            .dashboard-cards, .assignment-grid {
                grid-template-columns: 1fr;
            }
            .dashboard-cards > .dashboard-card:last-child:nth-child(odd) {
                grid-column: auto;
                width: 100%;
            }
            .assignment-card {
                min-height: 0;
                padding: 20px;
            }
        }
        .assignment-title {
            display: block;
            font-weight: bold;
            font-size: 18px;
            line-height: 1.35;
            margin-bottom: 4px;
        }
        /* Keep action buttons aligned even when an assignment title wraps. */
        .assignment-card .assignment-title {
            height: 2.7em;
            min-height: 2.7em;
            flex-shrink: 0;
        }
        /* Add a little space below the assignment cards before returning. */
        .container:has(.assignment-grid) > .home-btn {
            display: inline-block;
            margin-top: 26px;
        }
        .assignment-actions {
            display: flex;
            align-items: stretch;
            gap: 10px;
            width: 100%;
            margin-top: 12px;
        }
        .assignment-action-btn {
            flex: 1 1 0;
            min-width: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 11px 7px;
            background: #347fe5;
            color: white;
            text-align: center;
            border-radius: 6px;
            font-size: 13px;
            line-height: 1.3;
            overflow-wrap: anywhere;
        }
        .assignment-action-btn:hover {
            background: #2465c2;
            color: white;
            text-decoration: none;
        }
        .assignment-action-btn.view-btn {
            background: #673477;
        }
        .assignment-action-btn.view-btn:hover {
            background: #51285f;
        }
        .attempt-btn {
            display: block;
            width: 100%;
            padding: 11px;
            margin-top: 12px;
            background: #347fe5;
            color: white;
            text-align: center;
            border-radius: 6px;
        }
        .attempt-btn:hover {
            background: #2465c2;
            color: white;
            text-decoration: none;
        }
        .submitted-btn, .submitted-btn:hover {
            background: #23845b;
            cursor: default;
        }
        .logout {
            display: inline-block;
            margin-top: 20px;
            color: #dc2626;
        }
        .question-card {
            background: #f5f7fb;
            padding: 15px;
            margin: 15px 0;
            border-radius: 7px;
        }
        .back-link {
            display: block;
            text-align: center;
            margin-top: 20px;
        }
        .option-label {
            display: block;
            margin: 12px 0;
            color: #333;
        }
        .option-label input {
            width: auto;
            margin-right: 8px;
        }
        @media (max-width: 600px) {
            .container {
                padding: 22px;
            }
            body {
                padding: 15px;
            }
            .home-actions {
                gap: 12px;
            }
            .home-btn {
                width: 50%;
                padding: 12px 10px;
            }
        }
        @media (max-width: 480px) {
            .assignment-actions {
                flex-direction: column;
            }
        }
.submitted-code{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f4f4;padding:14px;border-radius:7px;border:1px solid #d8dce5;font-family:Consolas,monospace;font-size:13px;line-height:1.5;max-width:100%;overflow-x:auto}
.submitted-answers{display:flex;flex-direction:column;gap:12px}
.answer-item{padding:13px 15px;border:1px solid #d8dce5;border-radius:7px;background:#fff}
.answer-item p{margin:8px 0 0;line-height:1.55}
</style>
</head>
<body>
    <div class="container">
        <h1>IntelliGrade</h1>
        {{ content|safe }}
    </div>
</body>
</html>
"""
def show_page(content):
    return render_template_string(PAGE, content=content)
# HOME PAGE
@app.route("/")
def home():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return show_page("""
        <div class="welcome">
            <h2>Welcome to IntelliGrade</h2>
            <p>
                AI-Powered Assignment Evaluation
                and Automated Code Grading
            </p>
        </div>
        <div class="home-actions">
            <a class="home-btn" href="/login">Login</a>
            <a class="home-btn" href="/register">Register</a>
        </div>
    """)
# REGISTRATION
@app.route("/register", methods=["GET", "POST"])
def register():
    message = ""
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        role = request.form.get("role", "")
        if role not in ("teacher", "student"):
            message = "Invalid role."
        elif not name or not email or not password:
            message = "All fields are required."
        else:
            existing = execute_query(
                "SELECT user_id FROM users WHERE email = %s",
                (email,),
                True
            )
            if existing is None:
                message = "Database error. Please try again."
            elif existing:
                message = "Email already registered."
            else:
                hashed_password = generate_password_hash(password)
                result = execute_query(
                    """
                    INSERT INTO users
                    (name, email, password, role)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (name, email, hashed_password, role)
                )
                if result:
                    return redirect(url_for("login"))
                message = "Registration failed."
    return show_page("""
        <h2>Create Account</h2>
        <p class="message">""" + html.escape(message) + """</p>
        <form method="POST" autocomplete="off">
            <input name="name"
                   placeholder="Full Name"
                   autocomplete="off"
                   readonly onfocus="this.removeAttribute('readonly')"
                   required>
            <input type="email"
                   name="email"
                   placeholder="Email Address"
                   autocomplete="off"
                   readonly onfocus="this.removeAttribute('readonly')"
                   required>
            <input type="password"
                   name="password"
                   placeholder="Password"
                   autocomplete="new-password"
                   readonly onfocus="this.removeAttribute('readonly')"
                   required>
            <select name="role" required>
                <option value="" selected disabled>Select Role</option>
                <option value="student">Student</option>
                <option value="teacher">Teacher</option>
            </select>
            <button type="submit">Register</button>
        </form>
        <a class="page-link" href="/login">
            Already registered? Login
        </a>
    """)
# LOGIN
@app.route("/login", methods=["GET", "POST"])
def login():
    message = ""
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        users = execute_query(
            "SELECT * FROM users WHERE email = %s",
            (email,),
            True
        )
        if users and check_password_hash(
                        users[0]["password"], password
        ):
            session.clear()
            session["user_id"] = users[0]["user_id"]
            session["name"] = users[0]["name"]
            session["role"] = users[0]["role"]
            return redirect(url_for("dashboard"))
        message = "Invalid email or password."
    return show_page("""
        <h2>Login</h2>
        <p class="message">""" + html.escape(message) + """</p>
        <form method="POST" autocomplete="off">
            <input type="email"
                   name="email"
                   placeholder="Email Address"
                   autocomplete="on"
                   readonly onfocus="this.removeAttribute('readonly')"
                   required>
            <input type="password"
                   name="password"
                   placeholder="Password"
                   autocomplete="new-password"
                   readonly onfocus="this.removeAttribute('readonly')"
                   required>
            <button type="submit">Login</button>
        </form>
        <a class="page-link" href="/register">
            Create an account
        </a>
    """)
# DASHBOARD
@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))
    role = session.get("role")
    selected_view = request.args.get("view", "home")
    welcome = html.escape(str(session.get("name", "")))
    if role == "teacher":
        assignments = execute_query(
            """
            SELECT *
            FROM assignments
            WHERE teacher_id = %s
            ORDER BY assignment_id DESC
            """,
            (session["user_id"],),
            True
        ) or []
        if selected_view == "assignments":
            assignment_html = ""
            if assignments:
                for assignment in assignments:
                    assignment_id = assignment["assignment_id"]
                    title = html.escape(str(assignment["title"]))
                    assignment_type = html.escape(str(assignment["assignment_type"]))
                    if assignment["assignment_type"] == "programming":
                        add_link = url_for("manage_programming_assignment", assignment_id=assignment_id, mode="add")
                        view_link = url_for("manage_programming_assignment", assignment_id=assignment_id, mode="view")
                    else:
                        add_link = url_for("add_questions", assignment_id=assignment_id, mode="add")
                        view_link = url_for("add_questions", assignment_id=assignment_id, mode="view")
                    results_link = url_for("assignment_results", assignment_id=assignment_id)
                    edit_assignment_link = url_for("edit_assignment", assignment_id=assignment_id)
                    assignment_html += f"""
                    <div class="assignment-card">
                        <a class="assignment-title" href="{view_link}">{title}</a>
                        <p>Type: {assignment_type}</p>
                        <div class="assignment-actions">
                            <a class="assignment-action-btn" href="{add_link}">Add Questions</a>
                            <a class="assignment-action-btn view-btn" href="{view_link}">View Questions</a>
                            <a class="assignment-action-btn" href="{edit_assignment_link}">Edit Assignment</a>
                        </div>
                        <a class="page-link" href="{results_link}">View Student Results</a>
                    </div>
                    """
                assignment_html = "<div class='assignment-grid'>" + assignment_html + "</div>"
            else:
                assignment_html = "<p>No assignments created yet.</p>"
            content = f"""
                <h2>Your Assignments</h2>
                <p class="welcome">Select an assignment card to manage its questions and results.</p>
                {assignment_html}
                <a class="home-btn" href="/dashboard">Back to Dashboard</a>
            """
        else:
            count = len(assignments)
            content = f"""
                <h2>Teacher Dashboard</h2>
                <p class="welcome">Welcome, {welcome}</p>
                <div class="dashboard-cards">
                    <a class="dashboard-card" href="/create-assignment">
                        <strong>Create Assignment</strong>
                        <span>Create a new objective, descriptive, or programming assignment.</span>
                    </a>
                    <a class="dashboard-card" href="/dashboard?view=assignments">
                        <strong>Your Assignments</strong>
                        <span>{count} assignment(s) · Click to manage questions and results.</span>
                    </a>
                    <a class="dashboard-card" href="/analytics">
                        <strong>Performance Analytics</strong>
                        <span>Explore student performance, submissions, and marks.</span>
                    </a>
                    <a class="dashboard-card" href="/bulk-github-evaluation">
                        <strong>Bulk GitHub Evaluation</strong>
                        <span>Upload an Excel sheet of student GitHub repositories and evaluate code in bulk.</span>
                    </a>
                </div>
                <a class="logout" href="/logout">Logout</a>
            """
    else:
        assignments = execute_query(
            """
            SELECT assignment_id, title, description,
                   assignment_type, deadline, total_marks
            FROM assignments
            ORDER BY assignment_id DESC
            """,
            fetch=True
        ) or []
        if selected_view == "assignments":
            assignment_html = ""
            if assignments:
                for assignment in assignments:
                    assignment_id = assignment["assignment_id"]
                    title = html.escape(str(assignment["title"]))
                    description = html.escape(str(assignment["description"] or ""))
                    assignment_type = html.escape(str(assignment["assignment_type"]))
                    marks = html.escape(str(assignment["total_marks"]))
                    attempt_link = url_for("attempt_assignment", assignment_id=assignment_id)
                    previous_submission = execute_query(
                        "SELECT submission_id FROM submissions WHERE assignment_id=%s AND student_id=%s LIMIT 1",
                        (assignment_id, session["user_id"]), True
                    ) or []
                    already_submitted = bool(previous_submission)
                    expired = assignment_deadline_passed(assignment.get("deadline"))
                    deadline_text = (
                        html.escape(str(assignment.get("deadline")))
                        if assignment.get("deadline") else "No deadline"
                    )
                    if already_submitted:
                        action_html = f"<a class='attempt-btn submitted-btn' href='{url_for('my_submissions', assignment_id=assignment_id)}'>Already Submitted</a>"
                    elif expired:
                        action_html = "<span class='attempt-btn' style='background:#777;cursor:not-allowed'>Expired — Submission Closed</span>"
                    else:
                        action_html = f"<a class='attempt-btn' href='{attempt_link}'>Attempt Assignment</a>"
                    status_text = "Expired" if expired else "Open"
                    status_color = "#b42318" if expired else "#16804a"
                    assignment_html += f"""
                    <div class="assignment-card">
                        <span class="assignment-title">{title}</span>
                        <p>{description}</p>
                        <p>Type: {assignment_type}<br>Marks: {marks}</p>
                        <p>Deadline: {deadline_text}<br>Status: <strong style='color:{status_color}'>{status_text}</strong></p>
                        {action_html}
                    </div>
                    """
                assignment_html = "<div class='assignment-grid'>" + assignment_html + "</div>"
            else:
                assignment_html = "<p>No assignments available.</p>"
            content = f"""
                <h2>Available Assignments</h2>
                <p class="welcome">Choose an assignment card to start your attempt.</p>
                {assignment_html}
                <a class="home-btn" href="/dashboard">Back to Dashboard</a>
            """
        else:
            content = f"""
                <h2>Student Dashboard</h2>
                <p class="welcome">Welcome, {welcome}</p>
                <div class="dashboard-cards">
                    <a class="dashboard-card" href="/dashboard?view=assignments">
                        <strong>Available Assignments</strong>
                        <span>{len(assignments)} assignment(s) · Click to view and attempt.</span>
                    </a>
                    <a class="dashboard-card" href="/analytics">
                        <strong>My Performance Analytics</strong>
                        <span>View your marks, completed assignments, and progress.</span>
                    </a>
                    <a class="dashboard-card" href="/my-submissions">
                        <strong>My Submissions</strong>
                        <span>Review your submitted assignment history.</span>
                    </a>
                    <a class="dashboard-card" href="/student-feedback">
                        <strong>Automated Performance Feedback</strong>
                        <span>View your weak areas, improvement suggestions, and recommended practice.</span>
                    </a>
                </div>
                <a class="logout" href="/logout">Logout</a>
            """
    return show_page(content)
# PERFORMANCE ANALYTICS
@app.route("/analytics")
def performance_analytics():
    if "user_id" not in session:
        return redirect(url_for("login"))
    user_id = session["user_id"]
    role = session.get("role")
    selected_view = request.args.get("view", "overview")
    def metric(value):
        return html.escape(str(value if value is not None else 0))
    def make_card(label, value, view, hint):
        return (
            "<a class='metric-card' href='" + url_for("performance_analytics", view=view) + "'>"
            "<span>" + html.escape(label) + "</span>"
            "<strong>" + metric(value) + "</strong>"
            "<small>" + html.escape(hint) + "</small></a>"
        )
    if role == "teacher":
        stats_rows = execute_query("""
            SELECT
                (SELECT COUNT(*) FROM users WHERE role='student') AS total_students,
                (SELECT COUNT(*) FROM assignments WHERE teacher_id=%s) AS total_assignments,
                (SELECT COUNT(*) FROM submissions s JOIN assignments a ON a.assignment_id=s.assignment_id WHERE a.teacher_id=%s) AS total_submissions,
                (SELECT ROUND(AVG(r.marks_obtained),2) FROM results r JOIN assignments a ON a.assignment_id=r.assignment_id WHERE a.teacher_id=%s) AS average_marks,
                (SELECT ROUND(AVG(latest.marks_obtained / NULLIF(latest.total_marks,0) * 100),2)
                 FROM results latest
                 JOIN assignments a ON a.assignment_id=latest.assignment_id
                 WHERE a.teacher_id=%s
                   AND latest.result_id=(SELECT MAX(r2.result_id) FROM results r2
                                         WHERE r2.student_id=latest.student_id AND r2.assignment_id=latest.assignment_id)
                ) AS class_average_percentage,
                (SELECT COUNT(*) FROM (
                    SELECT r.student_id,
                           AVG(r.marks_obtained / NULLIF(r.total_marks,0) * 100) AS average_percentage
                    FROM results r
                    JOIN assignments a ON a.assignment_id=r.assignment_id
                    WHERE a.teacher_id=%s
                      AND r.result_id=(SELECT MAX(r2.result_id) FROM results r2
                                       WHERE r2.student_id=r.student_id AND r2.assignment_id=r.assignment_id)
                    GROUP BY r.student_id
                    HAVING average_percentage < 40
                ) AS at_risk) AS at_risk_students
        """, (user_id, user_id, user_id, user_id, user_id), True) or []
        stats = stats_rows[0] if stats_rows else {}
        assignment_rows = execute_query("""
            SELECT a.assignment_id, a.title, a.total_marks,
                   COUNT(r.result_id) AS evaluated_count,
                   ROUND(AVG(r.marks_obtained),2) AS average_marks,
                   MAX(r.marks_obtained) AS highest_marks,
                   MIN(r.marks_obtained) AS lowest_marks,
                   SUM(CASE WHEN r.marks_obtained >= a.total_marks*0.4 THEN 1 ELSE 0 END) AS passed,
                   SUM(CASE WHEN r.marks_obtained < a.total_marks*0.4 THEN 1 ELSE 0 END) AS failed
            FROM assignments a
            LEFT JOIN results r ON r.assignment_id=a.assignment_id
            WHERE a.teacher_id=%s
            GROUP BY a.assignment_id, a.title, a.total_marks
            ORDER BY a.assignment_id DESC
        """, (user_id,), True) or []
        student_rows = execute_query("""
            SELECT u.name, COUNT(*) AS assignments_completed,
                   ROUND(AVG(latest.marks_obtained),2) AS average_marks,
                   ROUND(AVG(latest.marks_obtained / NULLIF(latest.total_marks,0) * 100),2) AS average_percentage,
                   SUM(CASE WHEN latest.marks_obtained >= latest.total_marks*0.4 THEN 1 ELSE 0 END) AS passed,
                   SUM(CASE WHEN latest.marks_obtained < latest.total_marks*0.4 THEN 1 ELSE 0 END) AS failed
            FROM (
                SELECT r.student_id, r.assignment_id, r.marks_obtained, r.total_marks
                FROM results r
                JOIN (
                    SELECT student_id, assignment_id, MAX(result_id) AS latest_result_id
                    FROM results
                    GROUP BY student_id, assignment_id
                ) latest_ids ON latest_ids.latest_result_id=r.result_id
            ) latest
            JOIN assignments a ON a.assignment_id=latest.assignment_id
            JOIN users u ON u.user_id=latest.student_id
            WHERE a.teacher_id=%s
            GROUP BY u.user_id, u.name
            ORDER BY average_marks DESC
        """, (user_id,), True) or []
        submission_rows = execute_query("""
            SELECT s.submission_id, u.name AS student_name, a.title,
                   s.submission_type, r.marks_obtained, r.total_marks,
                   r.evaluation_status
            FROM submissions s
            JOIN assignments a ON a.assignment_id=s.assignment_id
            JOIN users u ON u.user_id=s.student_id
            LEFT JOIN results r ON r.submission_id=s.submission_id
            WHERE a.teacher_id=%s
              AND s.submission_id = (
                  SELECT MAX(s2.submission_id) FROM submissions s2
                  WHERE s2.assignment_id=s.assignment_id AND s2.student_id=s.student_id
              )
            ORDER BY s.submission_id DESC
        """, (user_id,), True) or []
        cards = [
            make_card("Total Students", stats.get("total_students", 0), "students", "Click to view student performance"),
            make_card("Your Assignments", stats.get("total_assignments", 0), "assignments", "Click to view assignment analytics"),
            make_card("Submissions", stats.get("total_submissions", 0), "submissions", "Click to view submitted work"),
            make_card("Class Average", str(stats.get("class_average_percentage") or 0) + "%", "students", "Average percentage across latest evaluated results"),
            make_card("Students Needing Support", stats.get("at_risk_students", 0), "at-risk", "Students averaging below 40%")
        ]
        labels = json.dumps([str(x["title"]) for x in assignment_rows])
        averages = json.dumps([float(x["average_marks"] or 0) for x in assignment_rows])
        pass_values = json.dumps([int(x["passed"] or 0) for x in assignment_rows])
        fail_values = json.dumps([int(x["failed"] or 0) for x in assignment_rows])
        assignment_table = "<div class='table-wrap'><table class='analytics-table'><tr><th>Assignment</th><th>Evaluated</th><th>Average</th><th>Highest</th><th>Lowest</th><th>Passed</th><th>Failed</th></tr>"
        for x in assignment_rows:
            assignment_table += "<tr><td>"+html.escape(str(x["title"]))+"</td><td>"+metric(x["evaluated_count"])+"</td><td>"+metric(x["average_marks"])+"</td><td>"+metric(x["highest_marks"])+"</td><td>"+metric(x["lowest_marks"])+"</td><td>"+metric(x["passed"])+"</td><td>"+metric(x["failed"])+"</td></tr>"
        assignment_table += "</table></div>" if assignment_rows else "<p>No evaluated assignment results yet.</p>"
        student_table = "<div class='table-wrap'><table class='analytics-table'><tr><th>Student</th><th>Assignments Completed</th><th>Average Marks</th><th>Average %</th><th>Passed</th><th>Failed</th><th>Performance</th></tr>"
        for x in student_rows:
            percentage = float(x.get("average_percentage") or 0)
            performance_status = "Needs Support" if percentage < 40 else ("Good" if percentage >= 75 else "Satisfactory")
            student_table += "<tr><td>"+html.escape(str(x["name"]))+"</td><td>"+metric(x["assignments_completed"])+"</td><td>"+metric(x["average_marks"])+"</td><td>"+metric(percentage)+"%</td><td>"+metric(x["passed"])+"</td><td>"+metric(x["failed"])+"</td><td>"+performance_status+"</td></tr>"
        student_table += "</table></div>" if student_rows else "<p>No student results available yet.</p>"
        at_risk_rows = [x for x in student_rows if float(x.get("average_percentage") or 0) < 40]
        at_risk_table = "<div class='table-wrap'><table class='analytics-table'><tr><th>Student</th><th>Assignments Completed</th><th>Average %</th><th>Failed Assignments</th><th>Suggested Action</th></tr>"
        for x in at_risk_rows:
            at_risk_table += "<tr><td>"+html.escape(str(x["name"]))+"</td><td>"+metric(x["assignments_completed"])+"</td><td>"+metric(x.get("average_percentage"))+"%</td><td>"+metric(x["failed"])+"</td><td>Arrange individual guidance and recommend practice</td></tr>"
        at_risk_table += "</table></div>" if at_risk_rows else "<p>No students are currently below the 40% support threshold.</p>"
        submission_table = "<div class='table-wrap'><table class='analytics-table'><tr><th>Submission ID</th><th>Student</th><th>Assignment</th><th>Type</th><th>Marks</th><th>Status</th></tr>"
        for x in submission_rows:
            score = str(x["marks_obtained"] if x["marks_obtained"] is not None else "Pending")
            if x["total_marks"] is not None:
                score += " / " + str(x["total_marks"])
            submission_table += "<tr><td>"+metric(x["submission_id"])+"</td><td>"+html.escape(str(x["student_name"]))+"</td><td>"+html.escape(str(x["title"]))+"</td><td>"+html.escape(str(x["submission_type"]))+"</td><td>"+html.escape(score)+"</td><td>"+html.escape(str(x["evaluation_status"] or "Pending"))+"</td></tr>"
        submission_table += "</table></div>" if submission_rows else "<p>No submissions received yet.</p>"
        if selected_view == "overview":
            detail = "<p class='analytics-hint'>Select any summary card to open its detailed analytics. Students averaging below 40% are flagged for support.</p>"
            content = "<h2>Teacher Performance Analytics</h2><p>Choose a card to explore the corresponding performance details.</p><div class='metrics-grid'>" + "".join(cards) + "</div>" + detail
        elif selected_view == "students":
            content = "<h2>Student Performance</h2><p>Performance summary using each student's latest evaluated result for each assignment.</p>" + student_table
        elif selected_view == "at-risk":
            content = "<h2>Students Needing Support</h2><p>Students whose average percentage across their latest evaluated assignment results is below 40%.</p>" + at_risk_table
        elif selected_view == "assignments":
            content = "<h2>Assignment Analytics</h2><p>Marks, evaluation counts, and pass/fail statistics for each assignment.</p><div class='analytics-grid'><div class='chart-card'><h3>Assignment-wise Average Marks</h3><canvas id='averageChart'></canvas></div><div class='chart-card'><h3>Pass / Fail Statistics</h3><canvas id='passChart'></canvas></div></div>" + assignment_table
        elif selected_view == "submissions":
            content = "<h2>Submission Details</h2><p>All student submissions for your assignments.</p>" + submission_table
        else:
            return redirect(url_for("performance_analytics"))
        if selected_view != "overview":
            content += "<a class='back-link' href='" + url_for("performance_analytics") + "'>Back to Analytics</a>"
        content += "<a class='page-link' href='/dashboard'>Back to Dashboard</a>"
        if selected_view == "assignments":
            content += "<script src='https://cdn.jsdelivr.net/npm/chart.js'></script><script>const labels=" + labels + ";"
            content += "const averages=" + averages + ";"
            content += "new Chart(document.getElementById('averageChart'),{type:'bar',data:{labels:labels,datasets:[{label:'Average marks',data:averages,backgroundColor:'#6750a4'}]},options:{responsive:true,scales:{y:{beginAtZero:true}}}});"
            content += "new Chart(document.getElementById('passChart'),{type:'bar',data:{labels:labels,datasets:[{label:'Passed',data:"+pass_values+",backgroundColor:'#269b65'},{label:'Failed',data:"+fail_values+",backgroundColor:'#d9534f'}]},options:{responsive:true,scales:{y:{beginAtZero:true}}}});"
            content += "</script>"
    else:
        stats_rows = execute_query("""
            SELECT COUNT(*) AS completed_assignments,
                   ROUND(AVG(latest.marks_obtained),2) AS average_marks,
                   MAX(latest.marks_obtained) AS highest_marks
            FROM (
                SELECT r.assignment_id, r.marks_obtained
                FROM results r
                JOIN (
                    SELECT assignment_id, student_id, MAX(result_id) AS latest_result_id
                    FROM results
                    WHERE student_id=%s
                    GROUP BY assignment_id, student_id
                ) latest_ids ON latest_ids.latest_result_id=r.result_id
            ) latest
        """, (user_id,), True) or []
        stats = stats_rows[0] if stats_rows else {}
        progress_rows = execute_query("""
            SELECT a.title, r.total_marks, r.marks_obtained,
                   r.evaluation_status, r.result_id, r.assignment_id
            FROM results r
            JOIN assignments a ON a.assignment_id=r.assignment_id
            JOIN (
                SELECT assignment_id, student_id, MAX(result_id) AS latest_result_id
                FROM results
                WHERE student_id=%s
                GROUP BY assignment_id, student_id
            ) latest_ids ON latest_ids.latest_result_id=r.result_id
            ORDER BY r.result_id
        """, (user_id,), True) or []
        cards = [
            make_card("Completed Assignments", stats.get("completed_assignments", 0), "completed", "Click to view completed assignments"),
            make_card("Average Marks", stats.get("average_marks") or 0, "average", "Click to view your performance")
        ]
        labels = json.dumps([str(x["title"]) for x in progress_rows])
        marks = json.dumps([float(x["marks_obtained"] or 0) for x in progress_rows])
        table = "<div class='table-wrap'><table class='analytics-table'><tr><th>Assignment</th><th>Marks Obtained</th><th>Total Marks</th><th>Status</th></tr>"
        for x in progress_rows:
            table += "<tr><td>"+html.escape(str(x["title"]))+"</td><td>"+metric(x["marks_obtained"])+"</td><td>"+metric(x["total_marks"])+"</td><td>"+html.escape(str(x["evaluation_status"]))+"</td></tr>"
        table += "</table></div>" if progress_rows else "<p>You have not submitted any assignments yet.</p>"
        if selected_view == "overview":
            content = "<h2>My Performance Analytics</h2><p>Select any card to view your detailed performance.</p><div class='metrics-grid'>" + "".join(cards) + "</div><p class='analytics-hint'>Your charts and result tables will appear after selecting a card.</p>"
        elif selected_view == "completed":
            content = "<h2>Completed Assignments</h2><p>Each assignment is listed once using its latest saved result.</p>" + table
        elif selected_view == "average":
            content = "<h2>Average Marks Analysis</h2><div class='average-summary-grid'><div class='average-summary'><span>Overall average marks</span><strong>"+metric(stats.get("average_marks"))+"</strong></div><div class='average-summary'><span>Highest marks</span><strong>"+metric(stats.get("highest_marks"))+"</strong></div></div><div class='chart-card student-chart-card'><h3>Assignment Performance Progress</h3><div class='student-chart-wrap'><canvas id='studentChart'></canvas></div></div>" + table
        else:
            return redirect(url_for("performance_analytics"))
        if selected_view != "overview":
            content += "<a class='back-link' href='" + url_for("performance_analytics") + "'>Back to Analytics</a>"
        content += "<a class='page-link' href='/dashboard'>Back to Dashboard</a>"
        if selected_view == "average":
            content += "<script src='https://cdn.jsdelivr.net/npm/chart.js'></script><script>new Chart(document.getElementById('studentChart'),{type:'line',data:{labels:"+labels+",datasets:[{label:'Marks obtained',data:"+marks+",borderColor:'#6750a4',backgroundColor:'rgba(103,80,164,.15)',fill:true,tension:.25}]},options:{responsive:true,maintainAspectRatio:false,scales:{y:{beginAtZero:true}}}});</script>"
    analytics_css = """
    <style>
    .container:has(.analytics-page){max-width:1540px;width:96%;padding:38px}
    .analytics-page{width:100%;max-width:1460px;margin:0 auto}
    .metrics-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:18px;margin:26px 0}
    .metric-card{background:#f5f3fb;border:1px solid #ded8ef;border-radius:12px;padding:22px;display:flex;flex-direction:column;gap:10px;text-decoration:none;min-height:145px;transition:transform .2s,box-shadow .2s}
    .metric-card:hover{transform:translateY(-4px);box-shadow:0 8px 22px rgba(30,20,70,.13);text-decoration:none;background:#eee9fa}
    .metric-card span{font-size:14px;color:#555}.metric-card strong{font-size:30px;color:#392b8b}.metric-card small{font-size:12px;color:#6750a4;line-height:1.4}
    .analytics-hint{padding:15px;background:#f7f5fc;border-radius:8px;color:#5b5270}
    .analytics-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px}
    .chart-card{background:#fff;border:1px solid #ddd;border-radius:12px;padding:20px;margin:18px 0;min-height:280px}
    .student-chart-card{min-height:0;max-width:760px;margin:18px auto;padding:16px 20px}
    .student-chart-wrap{position:relative;height:260px;max-height:260px}
    .student-chart-wrap canvas{max-height:260px}
    .average-summary-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin:20px 0}
    .average-summary{background:#f5f3fb;border:1px solid #ded8ef;border-radius:12px;padding:22px;margin:0;display:flex;justify-content:space-between;align-items:center;gap:15px;min-width:0}
    @media(max-width:650px){.average-summary-grid{grid-template-columns:1fr}}
    .average-summary strong{font-size:30px;color:#392b8b}
    .table-wrap{width:100%;overflow-x:auto}
    .analytics-table{width:100%;border-collapse:collapse;margin:14px 0 24px;font-size:14px;min-width:650px}
    .analytics-table th,.analytics-table td{border:1px solid #cfd2dc;padding:11px;text-align:left}
    .analytics-table th{background:#eeeaf7}.analytics-table tr:nth-child(even){background:#fafafa}
    .analytics-page .back-link{display:inline-block;margin:16px 10px 0 0;padding:10px 18px;border-radius:6px;background:#347fe5;color:white;text-align:center}
    .analytics-page .back-link:hover{background:#2465c2;text-decoration:none}
    @media(max-width:650px){.container:has(.analytics-page){padding:20px 14px}.analytics-page{overflow-x:auto}.metrics-grid{grid-template-columns:1fr 1fr}.metric-card{padding:15px}.metric-card strong{font-size:24px}}
    </style>
    """
    return show_page(analytics_css + "<div class='analytics-page'>" + content + "</div>")
# AUTOMATED STUDENT PERFORMANCE FEEDBACK
@app.route("/student-feedback")
def student_feedback():
    if "user_id" not in session:
        return redirect(url_for("login"))
    if session.get("role") != "student":
        return redirect(url_for("dashboard"))
    rows = execute_query("""
        SELECT a.title, a.assignment_type, r.marks_obtained, r.total_marks,
               r.evaluation_status, r.result_id, s.submission_type, s.answer_text
        FROM results r
        JOIN assignments a ON a.assignment_id=r.assignment_id
        LEFT JOIN submissions s ON s.submission_id=r.submission_id
        JOIN (
            SELECT assignment_id, student_id, MAX(result_id) AS latest_result_id
            FROM results
            WHERE student_id=%s
            GROUP BY assignment_id, student_id
        ) latest ON latest.latest_result_id=r.result_id
        ORDER BY r.result_id DESC
    """, (session["user_id"],), True) or []
    feedback_cards = []
    feedback_details = {}
    weak_topics = []
    total_percentage = 0.0
    evaluated_count = 0
    for row in rows:
        marks = float(row.get("marks_obtained") or 0)
        total = float(row.get("total_marks") or 0)
        if total <= 0:
            continue
        percentage = (marks / total) * 100
        total_percentage += percentage
        evaluated_count += 1
        assignment_type = str(row.get("assignment_type") or "").lower()
        title = html.escape(str(row.get("title") or "Assignment"))
        if percentage < 40:
            status = "Needs improvement"
            status_class = "feedback-warning"
            if "program" in assignment_type:
                topics = ["Python input and output", "Variables and data types", "Operators and program logic", "Testing with different inputs"]
                practice = "Practice small Python programs and run each program against multiple test inputs."
                explanation = "Your score indicates that the submitted solution did not satisfy enough evaluation marks. Review the assignment requirements and test your code with sample inputs."
            elif "descriptive" in assignment_type:
                topics = ["Understanding the question", "Including key concepts", "Organizing explanations", "Using relevant examples"]
                practice = "Rewrite the answer using the main concepts from the question and include a short example."
                explanation = "Your answer received a low evaluation score. Compare it with the question requirements and make sure the important concepts are explained clearly."
            else:
                topics = ["Reviewing question concepts", "Understanding options", "Revising incorrect responses"]
                practice = "Revisit the related lesson and solve a short set of similar questions."
                explanation = "Your marks indicate that some concepts need revision. Review the questions and concepts covered in this assignment."
            weak_topics.extend(topics)
        elif percentage < 75:
            status = "Making progress"
            status_class = "feedback-progress"
            topics = ["Revise concepts related to this assignment", "Practice similar questions", "Check mistakes before submission"]
            practice = "Complete a few additional practice questions and review any incorrect responses."
            explanation = "You have demonstrated partial understanding. More practice can help improve accuracy and confidence."
        else:
            status = "Good performance"
            status_class = "feedback-good"
            topics = ["Continue regular practice", "Try more challenging questions"]
            practice = "Attempt advanced exercises to strengthen your understanding."
            explanation = "You achieved a good result. Continue practising and try more challenging problems."
        topic_html = "".join("<li>"+html.escape(topic)+"</li>" for topic in topics)
        result_id = int(row.get("result_id") or 0)
        score_text = html.escape(str(round(marks,2)))+" / "+html.escape(str(round(total,2)))
        percentage_text = str(round(percentage,1))+"%"
        feedback_details[result_id] = (
            "<section class='feedback-detail'><div class='feedback-head'><h3>"+title+"</h3><span class='"+status_class+"'>"+status+"</span></div>"
            "<p><b>Score:</b> "+score_text+" ("+percentage_text+")</p><p>"+explanation+"</p><b>Topics to focus on</b><ul>"+topic_html+
            "</ul><p><b>Recommended practice:</b> "+practice+"</p></section>"
        )
        feedback_cards.append(
            "<a class='assignment-feedback-card' href='"+url_for('student_feedback', result_id=result_id)+"'>"
            "<div class='feedback-head'><h3>"+title+"</h3><span class='"+status_class+"'>"+status+"</span></div>"
            "<p><b>Score:</b> "+score_text+"</p><strong>"+percentage_text+"</strong><small>Click to view detailed feedback</small></a>"
        )
    if evaluated_count:
        overall_average = round(total_percentage / evaluated_count, 2)
        summary = "<div class='feedback-summary'><div><span>Evaluated assignments</span><strong>"+str(evaluated_count)+"</strong></div><div><span>Average percentage</span><strong>"+str(overall_average)+"%</strong></div></div>"
        if weak_topics:
            unique_topics = list(dict.fromkeys(weak_topics))
            focus_html = "".join("<li>"+html.escape(topic)+"</li>" for topic in unique_topics)
            focus = "<section class='feedback-card'><h3>Your priority improvement areas</h3><ul>"+focus_html+"</ul></section>"
        else:
            focus = "<section class='feedback-card'><h3>Keep up the good work</h3><p>No assignment is currently below the 40% improvement threshold. Continue practising to maintain your progress.</p></section>"
        selected_result_id = request.args.get("result_id", type=int)
        if selected_result_id is not None:
            content = "<h2>Assignment Feedback</h2>"
            if selected_result_id in feedback_details:
                content += feedback_details[selected_result_id]+"<a class='feedback-back' href='"+url_for('student_feedback')+"'>Back to all feedback</a>"
            else:
                content += "<p class='analytics-hint'>The selected feedback is not available.</p><a class='feedback-back' href='"+url_for('student_feedback')+"'>Back to all feedback</a>"
        else:
            content = "<h2>Automated Performance Feedback</h2><p>Feedback is generated from your latest saved marks and assignment type.</p>"+summary+focus+"<h3 class='feedback-section-title'>Assignment feedback</h3><div class='assignment-feedback-grid'>"+"".join(feedback_cards)+"</div>"
    else:
        content = "<h2>Automated Performance Feedback</h2><p>No evaluated results are available yet. Complete an assignment and check back to receive feedback.</p>"
    feedback_css = """
    <style>
    .feedback-summary{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin:22px 0}
    .feedback-summary div,.feedback-card,.feedback-detail{background:#f7f5fc;border:1px solid #ded8ef;border-radius:10px;padding:18px;margin:14px 0}
    .feedback-summary div{display:flex;justify-content:space-between;align-items:center;gap:12px}
    .feedback-summary span{color:#555}.feedback-summary strong{font-size:25px;color:#392b8b}
    .feedback-card h3,.feedback-detail h3{margin:0 0 12px;color:#29213f}.feedback-head{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}
    .feedback-warning,.feedback-progress,.feedback-good{padding:6px 10px;border-radius:15px;font-size:12px;font-weight:bold}
    .feedback-warning{background:#fff0e5;color:#a64b00}.feedback-progress{background:#fff8d9;color:#806000}.feedback-good{background:#e5f6ec;color:#176b3a}
    .feedback-card li,.feedback-detail li{margin:6px 0}
    .assignment-feedback-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:20px;margin:16px 0 24px;width:100%}
    .assignment-feedback-card{display:flex;flex-direction:column;gap:12px;background:#f7f5fc;border:1px solid #ded8ef;border-radius:12px;padding:20px;color:#29213f;text-decoration:none;transition:transform .2s,box-shadow .2s}
    .assignment-feedback-card:hover{transform:translateY(-3px);box-shadow:0 8px 20px rgba(30,20,70,.12);text-decoration:none;background:#f0ecfa}
    .assignment-feedback-card h3{margin:0;font-size:17px}.assignment-feedback-card strong{font-size:26px;color:#392b8b}
    .assignment-feedback-card small{color:#6750a4}.feedback-section-title{margin:24px 0 10px;color:#29213f}
    .feedback-back{display:inline-block;margin:10px 0;padding:10px 18px;border-radius:6px;background:#347fe5;color:white;text-decoration:none}
    .feedback-back:hover{background:#2465c2;text-decoration:none}
    @media(max-width:900px){.assignment-feedback-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
    @media(max-width:650px){.feedback-summary,.assignment-feedback-grid{grid-template-columns:1fr}.container:has(.analytics-page){width:100%;padding:20px 14px}}
    </style>
    """
    feedback_width_css = """
    <style>
    .container:has(.analytics-page .assignment-feedback-grid),
    .container:has(.analytics-page .feedback-detail) {
        width: calc(100% - 48px);
        max-width: 1320px;
        padding: 22px 48px;
    }
    .container:has(.analytics-page .assignment-feedback-grid) .analytics-page > h2,
    .container:has(.analytics-page .feedback-detail) .analytics-page > h2 {
        margin: 0 0 10px;
    }
    .container:has(.analytics-page .assignment-feedback-grid) .analytics-page > p {
        margin: 6px 0 10px;
    }
    .feedback-summary { margin: 10px 0; gap: 12px; }
    .feedback-summary div { padding: 12px 15px; margin: 0; }
    .feedback-card { padding: 12px 16px; margin: 10px 0; }
    .feedback-card h3 { margin-bottom: 7px; }
    .feedback-section-title { margin: 12px 0 7px; }
    .assignment-feedback-grid { gap: 14px; margin: 10px 0 8px; }
    .assignment-feedback-card { padding: 14px; gap: 8px; }
    .assignment-feedback-card h3 { font-size: 16px; }
    .assignment-feedback-card strong { font-size: 23px; }
    .feedback-detail { padding: 16px 18px; margin: 8px 0; }
    .feedback-detail p { margin: 8px 0; }
    .feedback-detail ul { margin: 6px 0; }
    .feedback-back { margin: 6px 0; padding: 8px 15px; }
    @media(max-width:900px) {
        .container:has(.analytics-page .assignment-feedback-grid),
        .container:has(.analytics-page .feedback-detail) {
            width: calc(100% - 32px);
            padding: 18px 24px;
        }
    }
    @media(max-width:650px) {
        .container:has(.analytics-page .assignment-feedback-grid),
        .container:has(.analytics-page .feedback-detail) {
            width: 100%;
            padding: 22px 16px;
        }
    }
    </style>
    """
    return show_page(feedback_width_css+feedback_css+"<div class='analytics-page'>"+content+"<a class='back-link' href='/dashboard'>Back to Dashboard</a></div>")
# CREATE ASSIGNMENT
@app.route("/create-assignment", methods=["GET", "POST"])
def create_assignment():
    if "user_id" not in session:
        return redirect(url_for("login"))
    if session["role"] != "teacher":
        return redirect(url_for("dashboard"))
    message = ""
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        description = request.form.get("description", "")
        assignment_type = request.form.get("assignment_type", "")
        deadline = request.form.get("deadline", "")
        total_marks = request.form.get("total_marks", "")
        if assignment_type not in (
            "objective", "descriptive", "programming"
        ):
            message = "Invalid assignment type."
        elif not title or not total_marks:
            message = "Title and total marks are required."
        else:
            try:
                marks_value = float(total_marks)
                if marks_value <= 0:
                    raise ValueError
            except ValueError:
                message = "Enter valid total marks."
            else:
                result = execute_query(
                    """
                    INSERT INTO assignments
                    (
                        teacher_id,
                        title,
                        description,
                        assignment_type,
                        deadline,
                        total_marks
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        session["user_id"],
                        title,
                        description,
                        assignment_type,
                        deadline if deadline else None,
                        marks_value
                    )
                )
                if result:
                    created = execute_query(
                        """
                        SELECT assignment_id
                        FROM assignments
                        WHERE teacher_id = %s
                        ORDER BY assignment_id DESC
                        LIMIT 1
                        """,
                        (session["user_id"],),
                        True
                    )
                    if created:
                        return redirect(
                            url_for(
                                "add_questions",
                                assignment_id=created[0]["assignment_id"],
                                mode="add"
                            )
                        )
                    return redirect(url_for("dashboard"))
                message = "Could not create assignment."
    return show_page("""
        <h2>Create Assignment</h2>
        <p class="message">""" + html.escape(message) + """</p>
        <form method="POST">
            <input name="title"
                   placeholder="Assignment Title"
                   required>
            <textarea name="description"
                      placeholder="Assignment Description"></textarea>
            <select name="assignment_type" required>
                <option value="objective">Objective</option>
                <option value="descriptive">Descriptive</option>
                <option value="programming">Programming</option>
            </select>
            <label>Deadline</label>
            <input type="datetime-local" name="deadline">
            <input type="number"
                   name="total_marks"
                   placeholder="Total Marks"
                   min="1"
                   step="0.01"
                   required>
            <button type="submit">
                Create Assignment
            </button>
        </form>
        <a class="page-link" href="/dashboard">
            Back to Dashboard
        </a>
    """)
# EDIT ASSIGNMENT DETAILS
@app.route("/assignment/<int:assignment_id>/edit", methods=["GET", "POST"])
def edit_assignment(assignment_id):
    if "user_id" not in session:
        return redirect(url_for("login"))
    if session.get("role") != "teacher":
        return redirect(url_for("dashboard"))
    rows = execute_query("SELECT * FROM assignments WHERE assignment_id=%s AND teacher_id=%s", (assignment_id, session["user_id"]), True) or []
    if not rows:
        return show_page("<h2>Assignment not found</h2><a class='back-link' href='/dashboard?view=assignments'>Back to Assignments</a>")
    assignment = rows[0]
    message = ""
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        description = request.form.get("description", "").strip()
        deadline = request.form.get("deadline", "").strip()
        marks_text = request.form.get("total_marks", "").strip()
        try:
            total_marks = float(marks_text)
            if not title or total_marks <= 0:
                raise ValueError
            updated = execute_query("UPDATE assignments SET title=%s, description=%s, deadline=%s, total_marks=%s WHERE assignment_id=%s AND teacher_id=%s", (title, description, deadline if deadline else None, total_marks, assignment_id, session["user_id"]))
            if updated:
                return redirect(url_for("dashboard", view="assignments", updated=1))
            message = "Could not update the assignment. Please try again."
        except (ValueError, TypeError):
            message = "Enter a valid title and positive total marks."
    deadline_value = assignment.get("deadline")
    if deadline_value:
        deadline_value = str(deadline_value).replace(" ", "T")[:16]
    else:
        deadline_value = ""
    current_status = "Expired" if assignment_deadline_passed(assignment.get("deadline")) else "Open"
    content = "<h2>Edit Assignment</h2><p class='message'>" + html.escape(message) + "</p><p>Current status: <strong>" + current_status + "</strong>. Change the deadline below to reopen the assignment when the new deadline is in the future.</p>"
    content += "<form method='post'><label>Assignment Title</label><input name='title' required value='" + html.escape(str(assignment.get("title", "")), quote=True) + "'><label>Description</label><textarea name='description' rows='4'>" + html.escape(str(assignment.get("description") or "")) + "</textarea><label>Assignment Type</label><input value='" + html.escape(str(assignment.get("assignment_type", "")), quote=True) + "' disabled><label>Deadline</label><input type='datetime-local' name='deadline' value='" + html.escape(deadline_value, quote=True) + "'><label>Total Marks</label><input type='number' name='total_marks' min='0.5' step='0.5' required value='" + html.escape(str(assignment.get("total_marks", "")), quote=True) + "'><button type='submit'>Save Changes</button></form><a class='back-link' href='/dashboard?view=assignments'>Cancel</a>"
    return show_page(content)
# ADD AND VIEW QUESTIONS
@app.route(
    "/assignment/<int:assignment_id>/questions",
    methods=["GET", "POST"]
)
def add_questions(assignment_id):
    if "user_id" not in session:
        return redirect(url_for("login"))
    if session["role"] != "teacher":
        return redirect(url_for("dashboard"))
    assignments = execute_query(
        "SELECT * FROM assignments WHERE assignment_id=%s AND teacher_id=%s",
        (assignment_id, session["user_id"]), True
    )
    if not assignments:
        return show_page("<h2>Assignment Not Found</h2><p class='message'>You are not authorized to access this assignment.</p><a class='page-link' href='/dashboard'>Back to Dashboard</a>")
    assignment = assignments[0]
    assignment_title = html.escape(str(assignment["title"]))
    assignment_type = assignment["assignment_type"]
    if assignment_type == "programming":
        return redirect(url_for("manage_programming_assignment", assignment_id=assignment_id))
    question_type = "mcq" if assignment_type == "objective" else "descriptive"
    mode = request.args.get("mode", "add")
    if mode not in ("add", "view"):
        mode = "add"
    message = ""
    edit_id = request.args.get("edit", type=int)
    selected = None
    if edit_id:
        selected_rows = execute_query(
            "SELECT * FROM questions WHERE question_id=%s AND assignment_id=%s",
            (edit_id, assignment_id), True
        ) or []
        if selected_rows:
            selected = selected_rows[0]
            mode = "add"
        else:
            message = "The selected question could not be found."
    if request.method == "POST":
        action = request.form.get("action", "add")
        question_id = request.form.get("question_id", type=int)
        question_text = request.form.get("question_text", "").strip()
        marks = request.form.get("marks", "").strip()
        option_a = request.form.get("option_a", "").strip()
        option_b = request.form.get("option_b", "").strip()
        option_c = request.form.get("option_c", "").strip()
        option_d = request.form.get("option_d", "").strip()
        correct_answer = request.form.get("correct_answer", "").strip().upper()
        reference_answer = request.form.get("reference_answer", "").strip()
        try:
            marks_value = float(marks)
            if marks_value <= 0 or not question_text:
                raise ValueError
        except (ValueError, TypeError):
            message = "Enter a valid question and positive marks."
        if not message and question_type == "mcq" and (not option_a or not option_b or not option_c or not option_d or correct_answer not in ("A", "B", "C", "D")):
            message = "Enter all four options and select the correct answer."
        if not message and question_type == "descriptive" and not reference_answer:
            message = "Enter the reference answer."
        if not message:
            total = execute_query(
                "SELECT COALESCE(SUM(marks),0) AS used_marks FROM questions WHERE assignment_id=%s AND (%s IS NULL OR question_id<>%s)",
                (assignment_id, question_id if action == "edit" else None, question_id if action == "edit" else None), True
            )
            if total is None:
                message = "Could not calculate existing question marks."
            elif float(total[0]["used_marks"] or 0) + marks_value > float(assignment["total_marks"] or 0):
                message = "Question marks exceed the assignment total marks."
        if not message:
            values = (
                question_text, question_type,
                option_a if question_type == "mcq" else None,
                option_b if question_type == "mcq" else None,
                option_c if question_type == "mcq" else None,
                option_d if question_type == "mcq" else None,
                correct_answer if question_type == "mcq" else None,
                reference_answer if question_type == "descriptive" else None,
                marks_value
            )
            if action == "edit" and question_id:
                result = execute_query(
                    "UPDATE questions SET question_text=%s, question_type=%s, option_a=%s, option_b=%s, option_c=%s, option_d=%s, correct_answer=%s, reference_answer=%s, marks=%s WHERE question_id=%s AND assignment_id=%s",
                    values + (question_id, assignment_id)
                )
                if result:
                    return redirect(url_for("add_questions", assignment_id=assignment_id, mode="view", updated=1))
                message = "Could not update the question."
            else:
                result = execute_query(
                    "INSERT INTO questions (assignment_id, question_text, question_type, option_a, option_b, option_c, option_d, correct_answer, reference_answer, marks) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (assignment_id,) + values
                )
                if result:
                    return redirect(url_for("add_questions", assignment_id=assignment_id, mode="view", added=1))
                message = "Could not save the question."
    questions = execute_query(
        "SELECT * FROM questions WHERE assignment_id=%s ORDER BY question_id ASC",
        (assignment_id,), True
    ) or []
    question_html = ""
    for count, question in enumerate(questions, start=1):
        question_html += (
            "<div class='question-card question-view-card'>"
            "<b>Question " + str(count) + ":</b> "
            "<strong>" + html.escape(str(question["question_text"])) + "</strong>"
            "<p><b>Marks:</b> " + html.escape(str(question["marks"])) + "</p>"
        )
        if question["question_type"] == "mcq":
            question_html += (
                "<div class='view-options'>"
                "<p>A. " + html.escape(str(question["option_a"] or "")) + "</p>"
                "<p>B. " + html.escape(str(question["option_b"] or "")) + "</p>"
                "<p>C. " + html.escape(str(question["option_c"] or "")) + "</p>"
                "<p>D. " + html.escape(str(question["option_d"] or "")) + "</p>"
                "</div><p><b>Correct answer:</b> "
                + html.escape(str(question["correct_answer"] or "")) + "</p>"
            )
        else:
            question_html += (
                "<p><b>Reference Answer:</b> "
                + html.escape(str(question["reference_answer"] or "")) + "</p>"
            )
        question_html += (
            "<a class='question-edit-link' href='"
            + url_for("add_questions", assignment_id=assignment_id, mode="add", edit=question["question_id"])
            + "'>Edit Question</a></div>"
        )
    if not question_html:
        question_html = "<p>No questions added yet.</p>"
    def safe_value(value):
        return html.escape(str(value or ""), quote=True)
    if question_type == "mcq":
        form_html = "<h3>" + ("Edit MCQ Question" if selected else "Add MCQ Question") + "</h3><form method='POST'><input type='hidden' name='action' value='" + ("edit" if selected else "add") + "'>"
        if selected:
            form_html += "<input type='hidden' name='question_id' value='" + str(selected["question_id"]) + "'>"
        form_html += "<textarea name='question_text' placeholder='Enter Question' required>" + (html.escape(str(selected["question_text"])) if selected else "") + "</textarea>"
        for key, label in (("option_a", "Option A"), ("option_b", "Option B"), ("option_c", "Option C"), ("option_d", "Option D")):
            form_html += "<input name='" + key + "' placeholder='" + label + "' required value='" + safe_value(selected.get(key) if selected else "") + "'>"
        form_html += "<label>Correct Answer</label><select name='correct_answer' required><option value=''>Select Correct Option</option>"
        for option in ("A", "B", "C", "D"):
            form_html += "<option value='" + option + "'" + (" selected" if selected and selected.get("correct_answer") == option else "") + ">Option " + option + "</option>"
        form_html += "</select><input type='number' name='marks' placeholder='Question Marks' min='0.01' step='0.01' required value='" + safe_value(selected.get("marks") if selected else "") + "'><button type='submit'>" + ("Update Question" if selected else "Add Question") + "</button></form>"
    else:
        form_html = "<h3>" + ("Edit Descriptive Question" if selected else "Add Descriptive Question") + "</h3><form method='POST'><input type='hidden' name='action' value='" + ("edit" if selected else "add") + "'>"
        if selected:
            form_html += "<input type='hidden' name='question_id' value='" + str(selected["question_id"]) + "'>"
        form_html += "<textarea name='question_text' placeholder='Enter Question' required>" + (html.escape(str(selected["question_text"])) if selected else "") + "</textarea><textarea name='reference_answer' placeholder='Enter Reference Answer' required>" + (html.escape(str(selected["reference_answer"] or "")) if selected else "") + "</textarea><input type='number' name='marks' placeholder='Question Marks' min='0.01' step='0.01' required value='" + safe_value(selected.get("marks") if selected else "") + "'><button type='submit'>" + ("Update Question" if selected else "Add Question") + "</button></form>"
    if selected:
        form_html += "<a class='back-link' href='" + url_for("add_questions", assignment_id=assignment_id, mode="view") + "'>Cancel Edit</a>"
    if mode == "view":
        page_heading = "View Questions"
        question_label = "Objective" if question_type == "mcq" else "Descriptive"
        main_section = (
            "<div class='question-view-page'>"
            "<h3>Added " + question_label + " Questions (" + str(len(questions)) + ")</h3>"
            + question_html
            + "</div>"
        )
    else:
        page_heading = "Edit Question" if selected else ("Add MCQ Questions" if question_type == "mcq" else "Add Descriptive Questions")
        main_section = form_html
    success = "Question added successfully!" if request.args.get("added") == "1" else ("Question updated successfully!" if request.args.get("updated") == "1" else "")
    view_styles = """<style>
.question-view-page{width:100%;max-width:100%;margin:0 auto}
.question-view-page h3{margin:18px 0 12px;font-size:18px;color:#222}
.question-view-card{background:#f8f7fc;border:1px solid #d8d8e8;border-radius:8px;padding:14px 15px;margin:10px 0 11px;color:#111}
.question-view-card p{margin:12px 0;color:#333;line-height:1.55}
.question-view-card strong{font-weight:700}
.question-view-card .view-options p{margin:5px 0 5px 12px}
.question-edit-link{display:block;text-align:center;margin:14px auto 2px;color:#2878df;text-decoration:none}
.question-view-page .home-btn{display:block;width:100%;max-width:230px;margin:18px auto 0}
</style>""" if mode == "view" else ""
    content = view_styles + "<h2>" + page_heading + "</h2><h3>" + assignment_title + "</h3><p>Assignment Type: " + html.escape(str(assignment_type)) + "</p><p>Total Marks: " + html.escape(str(assignment["total_marks"])) + "</p><p class='success'>" + success + "</p><p class='message'>" + html.escape(message) + "</p>" + main_section + "<a class='back-link' href='/dashboard'>Back to Dashboard</a>"
    return show_page(content)
# STUDENT ATTEMPT ASSIGNMENT
# Both URLs are supported: the attempt page displays questions,
# and the submit URL processes the student's answers.
@app.route(
    "/assignment/<int:assignment_id>/submit",
    methods=["POST"]
)
@app.route(
    "/assignment/<int:assignment_id>/attempt",
    methods=["GET", "POST"]
)
def attempt_assignment(assignment_id):
    if "user_id" not in session:
        return redirect(url_for("login"))
    if session["role"] != "student":
        return redirect(url_for("dashboard"))
    assignments = execute_query(
        "SELECT * FROM assignments WHERE assignment_id = %s",
        (assignment_id,),
        True
    )
    if not assignments:
        return show_page("""
            <h2>Assignment Not Found</h2>
            <a class="back-link" href="/dashboard">Back to Dashboard</a>
        """)
    assignment = assignments[0]
    if assignment_deadline_passed(assignment.get("deadline")):
        return show_page("<h2>Assignment Expired</h2><p>The deadline has passed. Submissions are closed for this assignment.</p><a class='back-link' href='/dashboard?view=assignments'>Back to Assignments</a>")
    existing_submission = execute_query(
        "SELECT submission_id FROM submissions WHERE assignment_id=%s AND student_id=%s LIMIT 1",
        (assignment_id, session["user_id"]), True
    ) or []
    if existing_submission:
        return show_page("<h2>Already Submitted</h2><p>You have already submitted this assignment. Only one submission is allowed.</p><a class='back-link' href='/my-submissions'>View My Submissions</a><a class='back-link' href='/dashboard'>Back to Dashboard</a>")
    if assignment["assignment_type"] == "programming":
        return redirect(url_for("attempt_programming_assignment", assignment_id=assignment_id))
    questions = execute_query(
        """
        SELECT * FROM questions
        WHERE assignment_id = %s
        ORDER BY question_id ASC
        """,
        (assignment_id,),
        True
    )
    if questions is None:
        return show_page("""
            <h2>Database Error</h2>
            <p>Unable to load assignment questions.</p>
            <a class="back-link" href="/dashboard">Back to Dashboard</a>
        """)
    if not questions:
        return show_page("""
            <h2>No Questions Available</h2>
            <p>The teacher has not added questions yet.</p>
            <a class="back-link" href="/dashboard">Back to Dashboard</a>
        """)
    if request.method == "POST":
        answers = {}
        obtained_marks = 0.0
        total_marks = 0.0
        descriptive_details = []
        for question in questions:
            question_id = question["question_id"]
            question_marks = float(question["marks"] or 0)
            total_marks += question_marks
            answer = request.form.get(
                "answer_" + str(question_id), ""
            ).strip()
            if not answer:
                return show_page("""
                    <h2>Submission Incomplete</h2>
                    <p class="message">Please answer every question.</p>
                    <a class="back-link" href="javascript:history.back()">
                        Return to Assignment
                    </a>
                """)
            answers[str(question_id)] = answer
            if question["question_type"] == "mcq":
                correct_answer = str(
                    question["correct_answer"] or ""
                ).strip().upper()
                if answer.upper() == correct_answer:
                    obtained_marks += question_marks
            elif question["question_type"] == "descriptive":
                reference = question.get("reference_answer") or ""
                marks, similarity = evaluate_descriptive_answer(
                    answer, reference, question_marks
                )
                obtained_marks += marks
                descriptive_details.append({
                    "question_id": question_id,
                    "marks": marks,
                    "max_marks": question_marks,
                    "similarity": similarity
                })
        duplicate_check = execute_query(
            "SELECT submission_id FROM submissions WHERE assignment_id=%s AND student_id=%s LIMIT 1",
            (assignment_id, session["user_id"]), True
        ) or []
        if duplicate_check:
            return show_page("<h2>Already Submitted</h2><p>Your submission has already been recorded.</p><a href='/my-submissions'>View My Submissions</a>")
        # Re-check immediately before saving in case the deadline passed while the page was open.
        latest_assignment = execute_query("SELECT deadline FROM assignments WHERE assignment_id=%s", (assignment_id,), True) or []
        if not latest_assignment or assignment_deadline_passed(latest_assignment[0].get("deadline")):
            return show_page("<h2>Assignment Expired</h2><p>The deadline has passed. Your answers were not submitted.</p><a class='back-link' href='/dashboard?view=assignments'>Back to Assignments</a>")
        answer_json = json.dumps(answers)
        saved = execute_query(
            """
            INSERT INTO submissions
            (assignment_id, student_id, submission_type, answer_text)
            VALUES (%s, %s, %s, %s)
            """,
            (
                assignment_id,
                session["user_id"],
                assignment["assignment_type"],
                answer_json
            )
        )
        if not saved:
            return show_page("""
                <h2>Submission Failed</h2>
                <p class="message">Could not save your submission.</p>
                <a class="back-link" href="/dashboard">Back to Dashboard</a>
            """)
        submission = execute_query(
            """
            SELECT submission_id
            FROM submissions
            WHERE assignment_id = %s AND student_id = %s
            ORDER BY submission_id DESC
            LIMIT 1
            """,
            (assignment_id, session["user_id"]),
            True
        )
        if not submission:
            return show_page("""
                <h2>Submission Saved</h2>
                <p>Your answers were saved, but the submission ID could not be retrieved.</p>
                <a class="back-link" href="/dashboard">Back to Dashboard</a>
            """)
        submission_id = submission[0]["submission_id"]
        obtained_marks = round(obtained_marks, 2)
        evaluation_status = "Evaluated"
        result_saved = execute_query(
            """
            INSERT INTO results
            (submission_id, student_id, assignment_id,
             marks_obtained, total_marks, evaluation_status)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                submission_id,
                session["user_id"],
                assignment_id,
                obtained_marks,
                total_marks,
                evaluation_status
            )
        )
        if not result_saved:
            return show_page("""
                <h2>Submission Saved</h2>
                <p>Your answers were saved, but the result could not be generated.</p>
                <a class="back-link" href="/dashboard">Back to Dashboard</a>
            """)
        if descriptive_details:
            detail_html = "<h3>Descriptive NLP Evaluation</h3>"
            for detail in descriptive_details:
                detail_html += (
                    "<p>Question " + str(detail["question_id"]) + ": "
                    + str(detail["marks"]) + " / "
                    + str(detail["max_marks"]) + " marks (text similarity: "
                    + str(detail["similarity"]) + "%)</p>"
                )
            result_message = (
                "<h3>Your Score: " + str(obtained_marks)
                + " / " + str(round(total_marks, 2)) + "</h3>"
                "<p class='success'>Your descriptive answers were evaluated using NLP text similarity.</p>"
                + detail_html
                + "<p>Please note: descriptive marks are generated automatically using NLP text similarity.</p>"
            )
        else:
            result_message = (
                "<h3>Your Score: " + str(obtained_marks)
                + " / " + str(round(total_marks, 2)) + "</h3>"
                "<p class='success'>Your MCQ answers were evaluated automatically.</p>"
            )
        return show_page("""
            <h2>Submission Successful</h2>
            <p class="success">Your answers have been submitted successfully!</p>
        """ + result_message + """
            <a class="back-link" href="/dashboard">Back to Dashboard</a>
        """)
    submit_url = url_for(
        "attempt_assignment",
        assignment_id=assignment_id
    )
    content = (
        "<h2>Attempt Assignment</h2>"
        "<h3>" + html.escape(str(assignment["title"])) + "</h3>"
        "<p>Type: " + html.escape(str(assignment["assignment_type"])) + "</p>"
        "<p>Total Marks: " + html.escape(str(assignment["total_marks"])) + "</p>"
        '<form method="POST" action="' + html.escape(submit_url) + '">'
    )
    for index, question in enumerate(questions, start=1):
        question_id = question["question_id"]
        content += (
            '<div class="question-card">'
            '<b>Question ' + str(index) + ':</b>'
            '<p>' + html.escape(str(question["question_text"])) + '</p>'
            '<p>Marks: ' + html.escape(str(question["marks"])) + '</p>'
        )
        if question["question_type"] == "mcq":
            options = [
                ("A", question["option_a"]),
                ("B", question["option_b"]),
                ("C", question["option_c"]),
                ("D", question["option_d"])
            ]
            for option_letter, option_text in options:
                content += (
                    '<label class="option-label">'
                    '<input type="radio" name="answer_' + str(question_id)
                    + '" value="' + option_letter + '" required>'
                    + option_letter + '. '
                    + html.escape(str(option_text or ""))
                    + '</label>'
                )
        else:
            content += (
                '<textarea name="answer_' + str(question_id)
                + '" placeholder="Enter your answer" required></textarea>'
            )
        content += '</div>'
    content += """
        <button type="submit">Submit Assignment</button>
        </form>
        <a class="back-link" href="/dashboard">Back to Dashboard</a>
    """
    return show_page(content)
# TEACHER VIEW STUDENT RESULTS
@app.route("/assignment/<int:assignment_id>/results")
@app.route("/assignment/<int:assignment_id>/results")
def assignment_results(assignment_id):
    if "user_id" not in session:
        return redirect(url_for("login"))
    if session.get("role") != "teacher":
        return redirect(url_for("dashboard"))
    assignment_rows = execute_query(
        "SELECT assignment_id, title FROM assignments WHERE assignment_id=%s AND teacher_id=%s",
        (assignment_id, session["user_id"]), True
    )
    if not assignment_rows:
        return show_page("<h2>Assignment Not Found</h2><a class='back-link' href='/dashboard'>Back to Dashboard</a>")
    rows = execute_query(
        """SELECT r.result_id, r.submission_id, r.student_id, u.name,
                  r.marks_obtained, r.total_marks, r.evaluation_status, r.evaluated_at,
                  s.submission_type, s.answer_text
           FROM results r
           JOIN users u ON u.user_id=r.student_id
           JOIN submissions s ON s.submission_id=r.submission_id
           WHERE r.assignment_id=%s
             AND r.result_id = (
                 SELECT MAX(r2.result_id) FROM results r2
                 WHERE r2.assignment_id=r.assignment_id AND r2.student_id=r.student_id
             )
           ORDER BY r.result_id DESC""",
        (assignment_id,), True
    )
    title = html.escape(str(assignment_rows[0]["title"]))
    content = "<div class='history-page'><h2>Assignment Submission History</h2><h3>" + title + "</h3>"
    if rows:
        content += "<p class='history-note'>Each student is shown once using their latest saved submission and result.</p>"
        content += "<div class='history-table-wrap'><table class='history-table'><thead><tr><th>Attempt</th><th>Student</th><th>Type</th><th>Marks</th><th>Status</th><th>Submitted</th><th>Details</th></tr></thead><tbody>"
        attempt_counts = {}
        for row in rows:
            sid = row["student_id"]
            attempt_counts[sid] = attempt_counts.get(sid, 0) + 1
            attempt_no = attempt_counts[sid]
            content += "<tr><td>" + str(attempt_no) + "</td><td class='student-cell'>" + html.escape(str(row["name"])) + "</td><td class='type-cell'>" + html.escape(str(row["submission_type"])) + "</td><td class='marks-cell'>" + html.escape(str(row["marks_obtained"])) + " / " + html.escape(str(row["total_marks"])) + "</td><td>" + html.escape(str(row["evaluation_status"])) + "</td><td class='date-cell'>" + html.escape(str(row["evaluated_at"] or "-")) + "</td><td class='details-cell'><a class='details-link' href='/submission/" + str(row["submission_id"]) + "'>View details</a></td></tr>"
        content += "</tbody></table></div>"
    elif rows is None:
        content += "<p class='message'>Unable to load submission history.</p>"
    else:
        content += "<p>No student submissions have been evaluated yet.</p>"
    content += "<a class='back-link' href='/dashboard'>Back to Dashboard</a></div>"
    return show_page(content)
@app.route("/my-submissions")
def my_submissions():
    if "user_id" not in session:
        return redirect(url_for("login"))
    if session.get("role") != "student":
        return redirect(url_for("dashboard"))
    selected_assignment_id = request.args.get("assignment_id", type=int)
    query = """SELECT r.result_id, r.submission_id, a.title, r.marks_obtained,
                      r.total_marks, r.evaluation_status, r.evaluated_at, s.submission_type
               FROM results r
               JOIN assignments a ON a.assignment_id=r.assignment_id
               JOIN submissions s ON s.submission_id=r.submission_id
               WHERE r.student_id=%s
                 AND (%s IS NULL OR r.assignment_id=%s)
                 AND r.result_id=(SELECT MAX(r2.result_id) FROM results r2
                                  WHERE r2.student_id=r.student_id AND r2.assignment_id=r.assignment_id)
               ORDER BY r.result_id DESC"""
    rows = execute_query(query, (session["user_id"], selected_assignment_id, selected_assignment_id), True)
    page_title = "My Submission History"
    if selected_assignment_id is not None:
        selected_assignment = execute_query("SELECT title FROM assignments WHERE assignment_id=%s", (selected_assignment_id,), True)
        if not selected_assignment:
            return show_page("<h2>Assignment not found</h2><a class='back-link' href='/dashboard'>Back to Dashboard</a>")
        page_title = "Result: " + html.escape(str(selected_assignment[0]["title"]))
    content = "<div class='my-submissions-page'><h2>" + page_title + "</h2>"
    if rows:
        content += "<div class='my-submissions-table-wrap'><table class='my-submissions-table'><thead><tr><th>Assignment</th><th>Type</th><th>Marks</th><th>Status</th><th>Date</th><th>Details</th></tr></thead><tbody>"
        for row in rows:
            content += "<tr><td>" + html.escape(str(row["title"])) + "</td><td>" + html.escape(str(row["submission_type"])) + "</td><td>" + html.escape(str(row["marks_obtained"])) + " / " + html.escape(str(row["total_marks"])) + "</td><td>" + html.escape(str(row["evaluation_status"])) + "</td><td>" + html.escape(str(row["evaluated_at"] or "-")) + "</td><td><a href='/submission/" + str(row["submission_id"]) + "'>View Details</a></td></tr>"
        content += "</tbody></table></div>"
    elif rows is None:
        content += "<p class='message'>Unable to load your submissions.</p>"
    else:
        content += "<p>No submissions found.</p>"
    content += "<a class='back-link' href='/dashboard'>Back to Dashboard</a></div>"
    return show_page(content)
@app.route("/submission/<int:submission_id>")
def submission_detail(submission_id):
    if "user_id" not in session:
        return redirect(url_for("login"))
    rows = execute_query(
        """SELECT s.submission_id, s.student_id, s.assignment_id, s.submission_type, s.answer_text,
                  r.marks_obtained, r.total_marks, r.evaluation_status, r.evaluated_at,
                  a.title, a.teacher_id, u.name
           FROM submissions s
           JOIN results r ON r.submission_id=s.submission_id
           JOIN assignments a ON a.assignment_id=s.assignment_id
           JOIN users u ON u.user_id=s.student_id
           WHERE s.submission_id=%s""",
        (submission_id,), True
    )
    if not rows:
        return show_page("<h2>Submission Not Found</h2><a href='/dashboard'>Back</a>")
    row = rows[0]
    is_owner = session.get("role") == "student" and session.get("user_id") == row["student_id"]
    is_teacher = session.get("role") == "teacher" and session.get("user_id") == row["teacher_id"]
    if not (is_owner or is_teacher):
        return show_page("<h2>Access Denied</h2><p>You are not allowed to view this submission.</p><a href='/dashboard'>Back</a>")
    content = "<div class='submission-page'><h2>Submission Details</h2>"
    content += "<div class='submission-meta'>"
    content += "<p><b>Assignment:</b> " + html.escape(str(row["title"])) + "</p>"
    content += "<p><b>Student:</b> " + html.escape(str(row["name"])) + "</p>"
    content += "<p><b>Type:</b> " + html.escape(str(row["submission_type"])) + "</p>"
    content += "<p><b>Marks:</b> " + html.escape(str(row["marks_obtained"])) + " / " + html.escape(str(row["total_marks"])) + "</p>"
    content += "<p><b>Status:</b> " + html.escape(str(row["evaluation_status"])) + "</p>"
    content += "<p><b>Evaluated:</b> " + html.escape(str(row["evaluated_at"] or "-")) + "</p></div>"
    submission_type = str(row["submission_type"] or "").lower()
    stored_answer = row["answer_text"] or ""
    if submission_type == "programming":
        content += "<div class='submission-section'><h3>Submitted Code</h3><pre class='submitted-code'>" + html.escape(str(stored_answer)) + "</pre></div>"
    else:
        try:
            submitted_answers = json.loads(stored_answer)
            if not isinstance(submitted_answers, dict):
                submitted_answers = {}
        except (ValueError, TypeError):
            submitted_answers = {}
        question_rows = execute_query(
            "SELECT question_id, question_text, question_type, option_a, option_b, option_c, option_d FROM questions WHERE assignment_id=%s",
            (row["assignment_id"],), True
        ) or []
        question_map = {str(q["question_id"]): q for q in question_rows}
        if submission_type == "objective":
            content += "<div class='submission-section'><h3>Submitted MCQ Options</h3>"
            if submitted_answers:
                for qid, selected in submitted_answers.items():
                    question = question_map.get(str(qid), {})
                    question_text = html.escape(str(question.get("question_text", "Question")))
                    selected_text = str(selected)
                    option_text = selected_text
                    option_key = selected_text.lower()
                    if question:
                        option_field = "option_" + option_key
                        if option_field in ("option_a", "option_b", "option_c", "option_d"):
                            option_text = question.get(option_field) or selected_text
                    content += "<div class='answer-item'><b>" + question_text + "</b><p><b>Selected option:</b> " + html.escape(selected_text.upper()) + " – " + html.escape(str(option_text)) + "</p></div>"
            else:
                content += "<p>No submitted options were found.</p>"
            content += "</div>"
        elif submission_type == "descriptive":
            content += "<div class='submission-section'><h3>Submitted Descriptive Answers</h3>"
            if submitted_answers:
                for qid, answer in submitted_answers.items():
                    question = question_map.get(str(qid), {})
                    question_text = html.escape(str(question.get("question_text", "Question")))
                    content += "<div class='answer-item'><b>" + question_text + "</b><p>" + html.escape(str(answer)) + "</p></div>"
            else:
                content += "<p>No submitted answers were found.</p>"
            content += "</div>"
        else:
            content += "<div class='submission-section'><h3>Submitted Answer</h3><pre class='submitted-code'>" + html.escape(str(stored_answer)) + "</pre></div>"
    if is_teacher:
        back_url = "/dashboard"
    else:
        back_url = "/my-submissions"
    content += "<a class='back-link' href='" + back_url + "'>Back</a></div>"
    return show_page(content)
# PROGRAMMING ASSIGNMENT MANAGEMENT AND AUTOMATIC CODE GRADING
@app.route("/assignment/<int:assignment_id>/programming", methods=["GET", "POST"])
def manage_programming_assignment(assignment_id):
    if "user_id" not in session:
        return redirect(url_for("login"))
    if session.get("role") != "teacher":
        return redirect(url_for("dashboard"))
    assignment_rows = execute_query("SELECT * FROM assignments WHERE assignment_id=%s AND teacher_id=%s AND assignment_type='programming'", (assignment_id, session["user_id"]), True)
    if not assignment_rows:
        return show_page("<h2>Programming assignment not found</h2><a href='/dashboard'>Back</a>")
    assignment = assignment_rows[0]
    mode = request.args.get("mode", "add")
    if mode not in ("add", "view"):
        mode = "add"
    message = ""
    edit_id = request.args.get("edit", type=int)
    if request.method == "POST":
        action = request.form.get("action", "add")
        question_id = request.form.get("question_id", type=int)
        title = request.form.get("question_title", "").strip()
        description = request.form.get("question_text", "").strip()
        marks_text = request.form.get("marks", "").strip()
        inputs = request.form.getlist("test_input[]")
        outputs = request.form.getlist("expected_output[]")
        try:
            marks = float(marks_text)
            if marks <= 0 or not title or not description:
                raise ValueError
            if len(inputs) != len(outputs):
                raise ValueError
            pairs = [(a.strip(), b.strip()) for a, b in zip(inputs, outputs) if a.strip() or b.strip()]
            if not pairs or any(not expected.strip() for _, expected in pairs):
                raise ValueError
            current = execute_query("SELECT programming_question_id, marks FROM programming_questions WHERE assignment_id=%s ORDER BY programming_question_id", (assignment_id,), True) or []
            if action == "edit":
                target = [q for q in current if int(q["programming_question_id"]) == int(question_id or 0)]
                if not target:
                    raise ValueError
                used_marks = sum(float(q["marks"] or 0) for q in current if int(q["programming_question_id"]) != int(question_id))
                if used_marks + marks > float(assignment["total_marks"]):
                    message = "Total programming question marks cannot exceed the assignment maximum marks."
                else:
                    updated = execute_query("UPDATE programming_questions SET question_title=%s, question_text=%s, marks=%s WHERE programming_question_id=%s AND assignment_id=%s", (title, description, marks, question_id, assignment_id))
                    if updated:
                        deleted = execute_query("DELETE FROM programming_test_cases WHERE programming_question_id=%s", (question_id,))
                        if deleted is None:
                            message = "Question details were updated, but old test cases could not be replaced."
                        else:
                            for test_input, expected in pairs:
                                if not execute_query("INSERT INTO programming_test_cases (programming_question_id, test_input, expected_output) VALUES (%s,%s,%s)", (question_id, test_input, expected)):
                                    message = "Question updated, but a test case could not be saved. Check the database."
                                    break
                        if not message:
                            return redirect(url_for("manage_programming_assignment", assignment_id=assignment_id, mode="view", updated=1))
                    else:
                        message = "Could not update the programming question."
            else:
                used_marks = sum(float(q["marks"] or 0) for q in current)
                if used_marks + marks > float(assignment["total_marks"]):
                    message = "Total programming question marks cannot exceed the assignment maximum marks."
                else:
                    qid = execute_insert("INSERT INTO programming_questions (assignment_id, question_title, question_text, marks) VALUES (%s,%s,%s,%s)", (assignment_id, title, description, marks))
                    if qid is None:
                        message = "Could not save the programming question. Check the programming_questions table and database connection."
                    else:
                        failed_case = False
                        for test_input, expected in pairs:
                            case_id = execute_insert("INSERT INTO programming_test_cases (programming_question_id, test_input, expected_output) VALUES (%s,%s,%s)", (qid, test_input, expected))
                            if case_id is None:
                                failed_case = True
                                break
                        if failed_case:
                            message = "The question was saved, but a test case could not be saved. Check the programming_test_cases table columns and database connection."
                        else:
                            return redirect(url_for("manage_programming_assignment", assignment_id=assignment_id, mode="view", saved=1))
        except (ValueError, TypeError):
            message = "Enter a valid title, question, positive marks, and at least one expected-output test case."
    questions = execute_query("SELECT * FROM programming_questions WHERE assignment_id=%s ORDER BY programming_question_id", (assignment_id,), True) or []
    selected = None
    if edit_id:
        selected_rows = execute_query("SELECT * FROM programming_questions WHERE programming_question_id=%s AND assignment_id=%s", (edit_id, assignment_id), True) or []
        if selected_rows:
            selected = selected_rows[0]
            mode = "add"
    title_text = html.escape(str(assignment["title"]))
    content = """<div class="programming-question-page"><style>
.programming-question-page h2{text-align:center;color:#2878e8;margin:0 0 18px}
.programming-question-page h3{margin:16px 0 12px}
.programming-question-page .assignment-meta{margin:12px 0;color:#333}
.programming-question-page form{margin-top:18px}
.programming-question-page label{display:block;font-weight:600;margin:12px 0 6px}
.programming-question-page input,.programming-question-page textarea{width:100%;padding:12px;border:1px solid #ccc;border-radius:5px;font-family:Arial,sans-serif;font-size:13px}
.programming-question-page textarea{resize:vertical;min-height:78px}
.programming-question-page button{width:100%;margin-top:14px;padding:11px;background:#347fe5;color:white;border:0;border-radius:5px;cursor:pointer}
.programming-question-page .test-case{border:1px solid #e0e3eb;border-radius:7px;padding:12px;margin:12px 0;background:#fafaff}
.programming-question-page .question-card{border:1px solid #d8d8e8;border-radius:8px;padding:15px;margin:12px 0;background:#f8f7fc}
.programming-question-page .back-link{display:block;text-align:center;margin-top:16px}
.programming-question-page .message,.programming-question-page .success{text-align:center}
</style><h2>""" + ("Add Programming Questions" if mode == "add" else "View Programming Questions") + """</h2><h3>""" + title_text + """</h3><p class="assignment-meta">Assignment Type: programming</p><p class="assignment-meta">Total Marks: """ + html.escape(str(assignment["total_marks"])) + """</p>"""
    if request.args.get("saved"):
        content += "<p class='success'>Programming question added successfully.</p>"
    if request.args.get("updated"):
        content += "<p class='success'>Programming question updated successfully.</p>"
    if message:
        content += "<p class='message'>" + html.escape(message) + "</p>"
    def field_value(value):
        return html.escape(str(value or ""), quote=True)
    if mode == "view" and not selected:
        content += "<h3>Added Programming Questions (" + str(len(questions)) + ")</h3>"
        if questions:
            for index, q in enumerate(questions, 1):
                cases = execute_query("SELECT test_case_id FROM programming_test_cases WHERE programming_question_id=%s", (q["programming_question_id"],), True) or []
                content += "<div class='question-card'><b>Question " + str(index) + ": " + html.escape(str(q["question_title"])) + "</b><p>" + html.escape(str(q["question_text"])) + "</p><p><b>Marks:</b> " + html.escape(str(q["marks"])) + "</p><p><b>Hidden test cases:</b> " + str(len(cases)) + "</p><a class='back-link' href='" + url_for("manage_programming_assignment", assignment_id=assignment_id, mode="add", edit=q["programming_question_id"]) + "'>Edit Question</a></div>"
        else:
            content += "<p>No programming questions added yet.</p>"
    else:
        form_title = "Edit Programming Question" if selected else "Add Programming Question"
        content += "<h3>" + form_title + "</h3>"
        old_cases = execute_query("SELECT test_input, expected_output FROM programming_test_cases WHERE programming_question_id=%s ORDER BY test_case_id", (selected["programming_question_id"],), True) or [] if selected else []
        content += "<form method='post'><input type='hidden' name='action' value='" + ("edit" if selected else "add") + "'>"
        if selected:
            content += "<input type='hidden' name='question_id' value='" + str(selected["programming_question_id"]) + "'>"
        content += "<label>Question Title</label><input name='question_title' placeholder='Enter Question Title' required value='" + field_value(selected["question_title"] if selected else "") + "'><label>Programming Question</label><textarea name='question_text' placeholder='Enter Programming Question' rows='4' required>" + html.escape(str(selected["question_text"] if selected else "")) + "</textarea><label>Question Marks</label><input type='number' name='marks' placeholder='Question Marks' min='0.01' step='0.01' required value='" + field_value(selected["marks"] if selected else "") + "'><h3>Hidden Test Cases</h3>"
        if old_cases:
            for index, (test_input, expected) in enumerate(old_cases, 1):
                content += "<div class='test-case'><b>Test Case " + str(index) + "</b><label>Input</label><textarea name='test_input[]' rows='2'>" + html.escape(str(test_input or "")) + "</textarea><label>Expected Output</label><textarea name='expected_output[]' rows='2' required>" + html.escape(str(expected or "")) + "</textarea></div>"
        else:
            for n in (1, 2):
                content += "<div class='test-case'><b>Test Case " + str(n) + "</b><label>Input</label><textarea name='test_input[]' rows='2' placeholder='Enter input (if required)'></textarea><label>Expected Output</label><textarea name='expected_output[]' rows='2' placeholder='Enter expected output'" + (" required" if n == 1 else "") + "></textarea></div>"
        content += "<button type='submit'>" + ("Update Question" if selected else "Add Question") + "</button></form>"
        if selected:
            content += "<a class='back-link' href='" + url_for("manage_programming_assignment", assignment_id=assignment_id, mode="view") + "'>Cancel Edit</a>"
    content += "<a class='back-link' href='/dashboard?view=assignments'>Back to Dashboard</a></div>"
    return show_page(content)
@app.route("/assignment/<int:assignment_id>/programming/attempt", methods=["GET", "POST"])
def attempt_programming_assignment(assignment_id):
    if "user_id" not in session:
        return redirect(url_for("login"))
    if session.get("role") != "student":
        return redirect(url_for("dashboard"))
    assignments = execute_query("SELECT * FROM assignments WHERE assignment_id=%s AND assignment_type='programming'", (assignment_id,), True)
    if not assignments:
        return show_page("<h2>Programming assignment not found</h2><a href='/dashboard'>Back</a>")
    assignment = assignments[0]
    if assignment_deadline_passed(assignment.get("deadline")):
        return show_page("<h2>Assignment Expired</h2><p>The deadline has passed. Programming submissions are closed.</p><a class='back-link' href='/dashboard?view=assignments'>Back to Assignments</a>")
    existing_submission = execute_query("SELECT submission_id FROM submissions WHERE assignment_id=%s AND student_id=%s LIMIT 1", (assignment_id, session["user_id"]), True) or []
    if existing_submission:
        return show_page("<h2>Already Submitted</h2><p>You have already submitted this programming assignment. Only one submission is allowed.</p><a class='back-link' href='/my-submissions'>View My Submissions</a><a class='back-link' href='/dashboard'>Back to Dashboard</a>")
    questions = execute_query("SELECT * FROM programming_questions WHERE assignment_id=%s ORDER BY programming_question_id", (assignment_id,), True) or []
    if not questions:
        return show_page("<h2>Not ready yet</h2><p>The teacher has not configured programming questions and test cases.</p><a href='/dashboard'>Back to Dashboard</a>")
    if request.method == "POST":
        submitted_codes = {}
        for question in questions:
            code = request.form.get("source_code_" + str(question["programming_question_id"]), "")
            if not code.strip():
                return show_page("<h2>Code required</h2><p>Please provide a solution for every programming question.</p><a href='javascript:history.back()'>Return</a>")
            submitted_codes[str(question["programming_question_id"])] = code
        total_obtained = 0.0
        total_question_marks = 0.0
        evaluation_details = []
        for question in questions:
            qid = question["programming_question_id"]
            cases = execute_query("SELECT test_input, expected_output FROM programming_test_cases WHERE programming_question_id=%s ORDER BY test_case_id", (qid,), True) or []
            if not cases:
                return show_page("<h2>Missing test cases</h2><p>Ask the teacher to configure test cases for every question.</p><a href='/dashboard'>Back</a>")
            passed = 0
            code = submitted_codes[str(qid)]
            for case in cases:
                result = run_python_test_case(code, str(case["test_input"] or ""))
                expected = str(case["expected_output"] or "").strip()
                actual = str(result["output"] or "").strip()
                if result["success"] and actual == expected:
                    passed += 1
            max_marks = float(question["marks"] or 0)
            obtained = round(max_marks * passed / len(cases), 2)
            total_obtained += obtained
            total_question_marks += max_marks
            evaluation_details.append({"title": str(question["question_title"]), "obtained": obtained, "maximum": max_marks, "passed": passed, "count": len(cases)})
        duplicate_check = execute_query("SELECT submission_id FROM submissions WHERE assignment_id=%s AND student_id=%s LIMIT 1", (assignment_id, session["user_id"]), True) or []
        if duplicate_check:
            return show_page("<h2>Already Submitted</h2><p>Your submission has already been recorded.</p><a href='/my-submissions'>View My Submissions</a>")
        # Re-check immediately before saving in case the deadline passed while coding.
        latest_assignment = execute_query("SELECT deadline FROM assignments WHERE assignment_id=%s", (assignment_id,), True) or []
        if not latest_assignment or assignment_deadline_passed(latest_assignment[0].get("deadline")):
            return show_page("<h2>Assignment Expired</h2><p>The deadline has passed. Your code was not submitted.</p><a class='back-link' href='/dashboard?view=assignments'>Back to Assignments</a>")
        stored_answer = json.dumps(submitted_codes)
        stored = execute_query("INSERT INTO submissions (assignment_id, student_id, submission_type, answer_text) VALUES (%s,%s,%s,%s)", (assignment_id, session["user_id"], "programming", stored_answer))
        if not stored:
            return show_page("<h2>Submission failed</h2><p>Could not save your code.</p><a href='/dashboard'>Back</a>")
        submission = execute_query("SELECT submission_id FROM submissions WHERE assignment_id=%s AND student_id=%s ORDER BY submission_id DESC LIMIT 1", (assignment_id, session["user_id"]), True)
        if not submission:
            return show_page("<h2>Submission saved</h2><p>Could not retrieve the submission ID.</p><a href='/dashboard'>Back</a>")
        assignment_total = float(assignment["total_marks"] or total_question_marks)
        saved_result = execute_query("INSERT INTO results (submission_id, student_id, assignment_id, marks_obtained, total_marks, evaluation_status) VALUES (%s,%s,%s,%s,%s,%s)", (submission[0]["submission_id"], session["user_id"], assignment_id, round(total_obtained, 2), assignment_total, "Evaluated"))
        if not saved_result:
            return show_page("<h2>Code submitted</h2><p>Evaluation ran, but result could not be saved.</p><a href='/dashboard'>Back</a>")
        content = "<h2>Programming Evaluation Result</h2><p>Total marks: <b>" + str(round(total_obtained, 2)) + " / " + str(assignment_total) + "</b></p><h3>Question-wise results</h3><table style='width:100%;border-collapse:collapse'><tr><th style='border:1px solid #ccc;padding:8px'>Question</th><th style='border:1px solid #ccc;padding:8px'>Marks</th><th style='border:1px solid #ccc;padding:8px'>Test cases</th></tr>"
        for detail in evaluation_details:
            content += "<tr><td style='border:1px solid #ccc;padding:8px'>" + html.escape(detail["title"]) + "</td><td style='border:1px solid #ccc;padding:8px'>" + str(detail["obtained"]) + " / " + str(detail["maximum"]) + "</td><td style='border:1px solid #ccc;padding:8px'>" + str(detail["passed"]) + " / " + str(detail["count"]) + " passed</td></tr>"
        content += "</table><a class='back-link' href='/my-submissions'>View My Submission History</a> <a class='back-link' href='/dashboard'>Back to Dashboard</a>"
        return show_page(content)
    content = "<h2>Programming Assignment</h2><h3>" + html.escape(str(assignment["title"])) + "</h3><p>Answer every programming question below. Your code will be evaluated separately for each question.</p>"
    for index, question in enumerate(questions, 1):
        content += "<div style='border:1px solid #ddd;border-radius:10px;padding:16px;margin:14px 0;background:#faf9fd'><h3>Question " + str(index) + ": " + html.escape(str(question["question_title"])) + "</h3><p>" + html.escape(str(question["question_text"])) + "</p><p><b>Marks:</b> " + html.escape(str(question["marks"])) + "</p><label>Python solution</label><textarea name='source_code_" + str(question["programming_question_id"]) + "' form='programming-submit-form' rows='12' required spellcheck='false' placeholder='Write your Python solution here'></textarea></div>"
    content += "<form id='programming-submit-form' method='post'><button type='submit'>Submit All Answers and Evaluate</button></form><p>Code runs in a restricted Docker container. Network access is disabled and execution time is limited.</p><a class='back-link' href='/dashboard'>Back to Dashboard</a>"
    return show_page(content)
# TEACHER BULK GITHUB CODE EVALUATION
@app.route("/bulk-github-evaluation", methods=["GET", "POST"])
def bulk_github_evaluation():
    if "user_id" not in session:
        return redirect(url_for("login"))
    if session.get("role") != "teacher":
        return redirect(url_for("dashboard"))
    teacher_id = session["user_id"]
    assignments = execute_query(
        """
        SELECT assignment_id, title, total_marks
        FROM assignments
        WHERE teacher_id=%s AND assignment_type='programming'
        ORDER BY assignment_id DESC
        """,
        (teacher_id,),
        True
    ) or []
    message = ""
    report_html = ""
    if request.method == "POST":
        assignment_id_text = request.form.get("assignment_id", "").strip()
        uploaded_file = request.files.get("excel_file")
        if not assignment_id_text.isdigit():
            message = "Please select a valid programming assignment."
        elif not uploaded_file or not uploaded_file.filename:
            message = "Please choose an Excel file."
        elif not uploaded_file.filename.lower().endswith((".xlsx", ".xls")):
            message = "Upload an .xlsx or .xls Excel file."
        else:
            selected = execute_query(
                """
                SELECT assignment_id, title, total_marks
                FROM assignments
                WHERE assignment_id=%s AND teacher_id=%s
                  AND assignment_type='programming'
                """,
                (int(assignment_id_text), teacher_id),
                True
            ) or []
            if not selected:
                message = "The selected programming assignment was not found."
            else:
                assignment = selected[0]
                try:
                    frame = pd.read_excel(uploaded_file)
                    frame.columns = [str(column).strip().lower() for column in frame.columns]
                    required_columns = {"student_id", "student_name", "github_url"}
                    if not required_columns.issubset(set(frame.columns)):
                        raise ValueError("Excel must contain student_id, student_name, and github_url columns.")
                    if len(frame.index) == 0:
                        raise ValueError("The Excel file has no student rows.")
                    if len(frame.index) > 100:
                        raise ValueError("A maximum of 100 student rows is allowed per upload.")
                    questions = execute_query(
                        """
                        SELECT programming_question_id, marks
                        FROM programming_questions
                        WHERE assignment_id=%s
                        ORDER BY programming_question_id
                        LIMIT 1
                        """,
                        (assignment["assignment_id"],),
                        True
                    ) or []
                    if not questions:
                        raise ValueError("This assignment does not have a programming question configured.")
                    question = questions[0]
                    test_cases = execute_query(
                        """
                        SELECT test_input, expected_output
                        FROM programming_test_cases
                        WHERE programming_question_id=%s
                        ORDER BY test_case_id
                        """,
                        (question["programming_question_id"],),
                        True
                    ) or []
                    if not test_cases:
                        raise ValueError("This programming question has no test cases configured.")
                    report_rows = []
                    for row_number, row in frame.iterrows():
                        raw_student_id = row.get("student_id")
                        raw_student_name = row.get("student_name")
                        github_url = str(row.get("github_url") or "").strip()
                        requested_file = str(row.get("file_path") or "").strip() if "file_path" in frame.columns else ""
                        try:
                            if pd.isna(raw_student_id) or pd.isna(raw_student_name):
                                raise ValueError("Student ID and student name are required.")
                            student_id_float = float(raw_student_id)
                            if not student_id_float.is_integer():
                                raise ValueError("Student ID must be a whole number.")
                            student_id = int(student_id_float)
                            student_name = str(raw_student_name).strip()
                            if not student_name or not github_url:
                                raise ValueError("Student name and GitHub URL are required.")
                            student_rows = execute_query(
                                "SELECT user_id, name FROM users WHERE user_id=%s AND role='student'",
                                (student_id,),
                                True
                            ) or []
                            if not student_rows:
                                raise ValueError("Student ID is not registered.")
                            if student_rows[0]["name"].strip().casefold() != student_name.casefold():
                                raise ValueError("Student name does not match the registered account.")
                            previous = execute_query(
                                "SELECT submission_id, answer_text FROM submissions WHERE assignment_id=%s AND student_id=%s ORDER BY submission_id DESC LIMIT 1",
                                (assignment["assignment_id"], student_id),
                                True
                            ) or []
                            if previous:
                                existing_result = execute_query(
                                    """
                                    SELECT marks_obtained, total_marks, evaluation_status
                                    FROM results
                                    WHERE assignment_id=%s AND student_id=%s
                                    ORDER BY result_id DESC
                                    LIMIT 1
                                    """,
                                    (assignment["assignment_id"], student_id),
                                    True
                                ) or []
                                saved_code = str(previous[0].get("answer_text") or "")
                                safe_repo = html.escape(github_url, quote=True)
                                repo_link = "<a href='" + safe_repo + "' target='_blank' rel='noopener noreferrer'>Open GitHub repository</a>"
                                safe_file = html.escape(requested_file or "main.py")
                                code_panel = "<details class='bulk-detail'><summary>View submitted Python code</summary><pre><code>" + html.escape(saved_code[:12000]) + ("\n... code preview truncated ..." if len(saved_code) > 12000 else "") + "</code></pre></details>"
                                if existing_result:
                                    saved_marks = float(existing_result[0]["marks_obtained"] or 0)
                                    saved_total = float(existing_result[0]["total_marks"] or assignment["total_marks"] or 0)
                                    saved_status = str(existing_result[0]["evaluation_status"] or "").strip()
                                    if saved_marks <= 0:
                                        display_status = "Failed"
                                    elif saved_total > 0 and saved_marks >= saved_total:
                                        display_status = "Passed"
                                    else:
                                        display_status = "Evaluated"
                                    detail_html = "<div><b>Repository:</b> " + repo_link + "</div><div><b>Python file:</b> " + safe_file + "</div><div><b>Saved evaluation:</b> " + html.escape(saved_status or display_status) + "</div><div class='bulk-note'>Per-test expected/actual outputs were not stored for this earlier evaluation, so the saved result is shown without rerunning the code.</div>" + code_panel
                                    report_rows.append(
                                        "<tr><td>" + html.escape(student_name) + "</td><td>" + display_status + " (Already Evaluated)</td><td>" + str(saved_marks) + " / " + str(saved_total) + "</td><td>" + detail_html + "</td></tr>"
                                    )
                                else:
                                    detail_html = "<div>Submission exists, but no saved evaluation result was found.</div><div><b>Repository:</b> " + repo_link + "</div><div><b>Python file:</b> " + safe_file + "</div>" + code_panel
                                    report_rows.append(
                                        "<tr><td>" + html.escape(student_name) + "</td><td>Already Submitted</td><td>—</td><td>" + detail_html + "</td></tr>"
                                    )
                                continue
                            source_code, selected_file = fetch_github_python_code(github_url, requested_file)
                            passed = 0
                            failures = []
                            case_detail_rows = []
                            for case_number, test_case in enumerate(test_cases, start=1):
                                evaluation = run_python_test_case(source_code, str(test_case.get("test_input") or ""))
                                expected_output = str(test_case.get("expected_output") or "").strip()
                                actual_output = str(evaluation.get("output") or "").strip()
                                case_passed = bool(evaluation.get("success") and actual_output == expected_output)
                                if case_passed:
                                    passed += 1
                                else:
                                    reason = evaluation.get("error") or "Output did not match expected output."
                                    failures.append("Test " + str(case_number) + ": " + str(reason)[:180])
                                case_status = "Passed" if case_passed else "Failed"
                                case_detail_rows.append("<tr><td>" + str(case_number) + "</td><td><pre>" + html.escape(expected_output[:1000]) + "</pre></td><td><pre>" + html.escape(actual_output[:1000] or "(no output)") + "</pre></td><td>" + case_status + ("<div>" + html.escape(str(evaluation.get("error") or "")[:300]) + "</div>" if not case_passed and evaluation.get("error") else "") + "</td></tr>")
                            question_marks = float(question["marks"] or 0)
                            obtained_marks = round(question_marks * passed / len(test_cases), 2)
                            total_marks = float(assignment["total_marks"] or question_marks)
                            submission_id = execute_insert(
                                """
                                INSERT INTO submissions
                                (assignment_id, student_id, submission_type, answer_text)
                                VALUES (%s,%s,%s,%s)
                                """,
                                (assignment["assignment_id"], student_id, "programming", source_code)
                            )
                            if not submission_id:
                                raise ValueError("Could not save the student's code submission.")
                            result_id = execute_insert(
                                """
                                INSERT INTO results
                                (submission_id, student_id, assignment_id, marks_obtained, total_marks, evaluation_status)
                                VALUES (%s,%s,%s,%s,%s,%s)
                                """,
                                (submission_id, student_id, assignment["assignment_id"], obtained_marks, total_marks, "Evaluated")
                            )
                            if not result_id:
                                report_rows.append(
                                    "<tr><td>" + html.escape(student_name) + "</td><td>Saved submission; result could not be saved</td><td>—</td><td>" + html.escape(selected_file) + "</td></tr>"
                                )
                                continue
                            status_text = str(passed) + " / " + str(len(test_cases)) + " tests passed"
                            safe_repo = html.escape(github_url, quote=True)
                            detail_html = "<div><b>Repository:</b> <a href='" + safe_repo + "' target='_blank' rel='noopener noreferrer'>Open GitHub repository</a></div><div><b>Python file:</b> " + html.escape(selected_file) + "</div><div><b>Test case breakdown:</b></div><div class='bulk-detail-table-wrap'><table class='bulk-detail-table'><thead><tr><th>#</th><th>Expected output</th><th>Actual output</th><th>Status / Error</th></tr></thead><tbody>" + "".join(case_detail_rows) + "</tbody></table></div><details class='bulk-detail'><summary>View submitted Python code</summary><pre><code>" + html.escape(source_code[:12000]) + ("\n... code preview truncated ..." if len(source_code) > 12000 else "") + "</code></pre></details>"
                            report_rows.append(
                                "<tr><td>" + html.escape(student_name) + "</td><td>" + html.escape(status_text) + "</td><td>" + str(obtained_marks) + " / " + str(total_marks) + "</td><td>" + detail_html + "</td></tr>"
                            )
                        except Exception as student_error:
                            report_rows.append(
                                "<tr><td>" + html.escape(str(raw_student_name) if not pd.isna(raw_student_name) else "Row " + str(row_number + 2)) + "</td><td>Skipped / Failed</td><td>—</td><td>" + html.escape(str(student_error)) + "</td></tr>"
                            )
                    report_html = (
                        "<section class='bulk-report'>"
                        "<h3>Bulk Evaluation Report</h3>"
                        "<div class='bulk-report-table-wrap'><table class='bulk-report-table'>"
                        "<thead><tr><th>Student</th><th>Test Results</th><th>Marks</th><th>Details</th></tr></thead>"
                        "<tbody>" + "".join(report_rows) + "</tbody>"
                        "</table></div></section>"
                    )
                    message = "Excel processing completed. Opening the report..."
                    report_token = uuid.uuid4().hex
                    BULK_REPORT_CACHE[report_token] = {"html": report_html, "assignment_title": str(assignment["title"]), "created_by": teacher_id}
                    session["bulk_report_token"] = report_token
                    return redirect(url_for("bulk_github_report", report_id=report_token))
                except Exception as error:
                    message = str(error)
    assignment_options = "<option value=''>Select programming assignment</option>"
    for item in assignments:
        assignment_options += (
            "<option value='" + str(item["assignment_id"]) + "'>"
            + html.escape(str(item["title"])) + " (ID: "
            + str(item["assignment_id"]) + ")</option>"
        )
    content = """
        <div class="bulk-github-page">
        <h2>Bulk GitHub Evaluation</h2>
        <p>Upload an Excel sheet containing student details and public GitHub repository links. IntelliGrade fetches the Python solution and evaluates it using the selected assignment's existing Docker test cases.</p>
        <p><b>Required Excel columns:</b> student_id, student_name, github_url</p>
        <p><b>Optional column:</b> file_path (use this when the repository contains multiple Python files).</p>
        <p class="message">""" + html.escape(message) + """</p>
        <form method="POST" enctype="multipart/form-data">
            <label>Select Programming Assignment</label>
            <select name="assignment_id" required>""" + assignment_options + """</select>
            <label>Upload Excel File</label>
            <input type="file" name="excel_file" accept=".xlsx,.xls" required>
            <button type="submit">Upload and Evaluate</button>
        </form>
        """ + report_html + """
        <a class="back-link" href="/dashboard">Back to Dashboard</a>
        </div>
    """
    return show_page(content)
@app.route("/bulk-github-report")
def bulk_github_report():
    if "user_id" not in session or session.get("role") != "teacher":
        return redirect(url_for("login"))
    report_id = request.args.get("report_id", "")
    if not report_id or report_id != session.get("bulk_report_token"):
        return redirect(url_for("bulk_github_evaluation"))
    report = BULK_REPORT_CACHE.get(report_id)
    if not report or report.get("created_by") != session.get("user_id"):
        return redirect(url_for("bulk_github_evaluation"))
    content = "<div class='bulk-github-page'><h2>Bulk Evaluation Report</h2><p><b>Assignment:</b> " + html.escape(report["assignment_title"]) + "</p><div class='bulk-report-actions'><a class='bulk-download-button' href='" + url_for("download_bulk_github_report_pdf", report_id=report_id) + "'>Download PDF Report</a><a class='bulk-secondary-button' href='" + url_for("bulk_github_evaluation") + "'>Upload Another Excel</a></div>" + report["html"] + "<a class='back-link' href='" + url_for("dashboard") + "'>Back to Dashboard</a></div>"
    return show_page(content)
class BulkReportHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.links = []
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ("br", "div", "p", "pre", "tr", "summary"):
            self.parts.append("\n")
        if tag == "a" and attrs.get("href"):
            self.links.append(attrs["href"])
            self.parts.append(" [Link: " + attrs["href"] + "] ")
    def handle_endtag(self, tag):
        if tag in ("div", "p", "pre", "tr", "details", "summary"):
            self.parts.append("\n")
    def handle_data(self, data):
        self.parts.append(data)
    def get_text(self):
        return re.sub(r"[ \t]+", " ", "".join(self.parts)).strip()
@app.route("/download-bulk-github-report/<report_id>.pdf")
def download_bulk_github_report_pdf(report_id):
    if "user_id" not in session or session.get("role") != "teacher":
        return redirect(url_for("login"))
    report = BULK_REPORT_CACHE.get(report_id)
    if not report or report_id != session.get("bulk_report_token") or report.get("created_by") != session.get("user_id"):
        return redirect(url_for("bulk_github_evaluation"))
    row_matches = re.findall(r"<tr[^>]*>(.*?)</tr>", report["html"], flags=re.I | re.S)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=17, leading=21, alignment=TA_CENTER, textColor=colors.HexColor("#2878df"), spaceAfter=8)
    sub_style = ParagraphStyle("ReportSub", parent=styles["Normal"], fontSize=9, leading=12, spaceAfter=8)
    cell_style = ParagraphStyle("ReportCell", parent=styles["BodyText"], fontName="Helvetica", fontSize=7, leading=9, wordWrap="CJK")
    heading_style = ParagraphStyle("StudentHeading", parent=styles["Heading3"], fontSize=10, leading=13, textColor=colors.HexColor("#263247"), spaceBefore=8, spaceAfter=4)
    output = io.BytesIO()
    pdf = SimpleDocTemplate(output, pagesize=landscape(A4), rightMargin=12*mm, leftMargin=12*mm, topMargin=12*mm, bottomMargin=14*mm, title="IntelliGrade Bulk Evaluation Report")
    story = [Paragraph("IntelliGrade – Bulk Evaluation Report", title_style), Paragraph("Assignment: " + html.escape(str(report.get("assignment_title", ""))), sub_style), Paragraph("Generated: " + html.escape(pd.Timestamp.now().strftime("%d-%m-%Y %H:%M")), sub_style), Spacer(1, 4)]
    header = [Paragraph("Student", cell_style), Paragraph("Test Results", cell_style), Paragraph("Marks", cell_style), Paragraph("Submission Details", cell_style)]
    table_data = [header]
    for table_row in row_matches:
        cells = re.findall(r"<td[^>]*>(.*?)</td>", table_row, flags=re.I | re.S)
        if len(cells) != 4:
            continue
        parsed_cells = []
        for cell in cells:
            parser = BulkReportHTMLParser()
            parser.feed(cell)
            text_value = parser.get_text()
            safe_text = html.escape(text_value).replace("\n", "<br/>")
            parsed_cells.append(Paragraph(safe_text or "—", cell_style))
        table_data.append(parsed_cells)
    if len(table_data) == 1:
        story.append(Paragraph("No student evaluation rows were recorded.", sub_style))
    else:
        report_table = Table(table_data, colWidths=[43*mm, 43*mm, 30*mm, 145*mm], repeatRows=1, hAlign="LEFT")
        report_table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#eaf0fa")), ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#263247")), ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#b9c5d6")), ("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 6), ("RIGHTPADDING", (0,0), (-1,-1), 6), ("TOPPADDING", (0,0), (-1,-1), 6), ("BOTTOMPADDING", (0,0), (-1,-1), 6), ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f8faff")])]))
        story.append(report_table)
    def add_page_number(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawRightString(landscape(A4)[0] - 12*mm, 7*mm, "Page " + str(document.page))
        canvas.restoreState()
    pdf.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    output.seek(0)
    filename = "IntelliGrade_Bulk_Evaluation_Report.pdf"
    return send_file(output, as_attachment=True, download_name=filename, mimetype="application/pdf")
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))
# RUN APPLICATION
if __name__ == "__main__":
    app.run(
        debug=True,
        port=5001
    )