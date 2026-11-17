"""
PHASE 15 — SECURITY TESTING AND AUDIT

Tests the local Medical AI Platform API.
Does not use real patient information.

Usage:
    python AIML-02/phase15_security_testing.py

    python AIML-02/phase15_security_testing.py \
        --base-url http://127.0.0.1:8000
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "AIML-02"
    / "data"
    / "security"
)

REPORT_PATH = OUTPUT_DIR / "phase15_report.json"

ALLOWED_FRONTEND = "http://localhost:5173"

TIMEOUT = 15


class SecurityTester:

    def __init__(self, base_url):

        self.base_url = base_url.rstrip("/")
        self.results = []

        self.session = requests.Session()

        self.session.headers.update({
            "User-Agent": "MedicalAI-Security-Audit/1.0"
        })

    # --------------------------------------------------------
    # RECORD RESULTS
    # --------------------------------------------------------

    def record(self, name, status, details, severity="INFO"):

        result = {
            "test": name,
            "status": status,
            "severity": severity,
            "details": details,
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        self.results.append(result)

        print(
            f"[{status}] {name}: {details}"
        )

    # --------------------------------------------------------
    # HTTP REQUEST HELPER
    # --------------------------------------------------------

    def request(self, method, path, **kwargs):

        return self.session.request(
            method,
            self.base_url + path,
            timeout=TIMEOUT,
            **kwargs,
        )

    # --------------------------------------------------------
    # 1. BACKEND HEALTH
    # --------------------------------------------------------

    def test_backend_health(self):

        try:

            response = self.request(
                "GET",
                "/health",
            )

            if response.status_code == 200:

                self.record(
                    "Backend health",
                    "PASS",
                    "Backend health endpoint returned HTTP 200.",
                )

            else:

                self.record(
                    "Backend health",
                    "FAIL",
                    f"Unexpected HTTP {response.status_code}.",
                    "HIGH",
                )

        except requests.RequestException as error:

            self.record(
                "Backend health",
                "FAIL",
                str(error),
                "HIGH",
            )

    # --------------------------------------------------------
    # 2. DOCTOR REVIEW HEALTH
    # --------------------------------------------------------

    def test_doctor_health(self):

        try:

            response = self.request(
                "GET",
                "/api/v1/doctor/health",
            )

            if response.status_code == 200:

                self.record(
                    "Doctor review health",
                    "PASS",
                    "Doctor review service is responding.",
                )

            else:

                self.record(
                    "Doctor review health",
                    "FAIL",
                    f"HTTP {response.status_code}.",
                    "HIGH",
                )

        except requests.RequestException as error:

            self.record(
                "Doctor review health",
                "FAIL",
                str(error),
                "HIGH",
            )

    # --------------------------------------------------------
    # 3. INVALID FILE UPLOAD
    # --------------------------------------------------------

    def test_invalid_upload(self):

        try:

            response = self.request(
                "POST",
                "/api/v1/predict",
                files={
                    "file": (
                        "invalid.txt",
                        b"This is not a valid medical image.",
                        "text/plain",
                    )
                },
                data={
                    "explain": "false"
                },
            )

            if response.status_code in {
                400,
                413,
                415,
                422,
            }:

                self.record(
                    "Invalid file rejection",
                    "PASS",
                    f"Invalid upload rejected with HTTP {response.status_code}.",
                )

            elif response.status_code >= 500:

                self.record(
                    "Invalid file rejection",
                    "FAIL",
                    f"Server error HTTP {response.status_code}.",
                    "HIGH",
                )

            else:

                self.record(
                    "Invalid file rejection",
                    "FAIL",
                    f"Unexpected HTTP {response.status_code}.",
                    "HIGH",
                )

        except requests.RequestException as error:

            self.record(
                "Invalid file rejection",
                "FAIL",
                str(error),
                "HIGH",
            )

    # --------------------------------------------------------
    # 4. MISSING FILE VALIDATION
    # --------------------------------------------------------

    def test_missing_upload(self):

        try:

            response = self.request(
                "POST",
                "/api/v1/predict",
                data={
                    "explain": "false"
                },
            )

            if response.status_code in {
                400,
                422,
            }:

                self.record(
                    "Missing file validation",
                    "PASS",
                    "API rejected request without an uploaded file.",
                )

            else:

                self.record(
                    "Missing file validation",
                    "WARN",
                    f"Received HTTP {response.status_code}.",
                    "MEDIUM",
                )

        except requests.RequestException as error:

            self.record(
                "Missing file validation",
                "FAIL",
                str(error),
                "HIGH",
            )

    # --------------------------------------------------------
    # 5. CORS VALIDATION
    # --------------------------------------------------------

    def test_cors(self):

        try:

            allowed_response = self.request(
                "OPTIONS",
                "/api/v1/predict",
                headers={
                    "Origin": ALLOWED_FRONTEND,
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "content-type",
                },
            )

            allowed_origin = (
                allowed_response.headers.get(
                    "access-control-allow-origin"
                )
            )

            if allowed_origin == ALLOWED_FRONTEND:

                self.record(
                    "Allowed CORS origin",
                    "PASS",
                    "Configured frontend origin is allowed.",
                )

            else:

                self.record(
                    "Allowed CORS origin",
                    "WARN",
                    f"Unexpected allowed origin: {allowed_origin}",
                    "MEDIUM",
                )

            denied_response = self.request(
                "OPTIONS",
                "/api/v1/predict",
                headers={
                    "Origin": "https://untrusted.example",
                    "Access-Control-Request-Method": "POST",
                },
            )

            denied_origin = (
                denied_response.headers.get(
                    "access-control-allow-origin"
                )
            )

            if denied_origin is None:

                self.record(
                    "Untrusted CORS origin",
                    "PASS",
                    "Untrusted origin was not granted CORS access.",
                )

            else:

                self.record(
                    "Untrusted CORS origin",
                    "FAIL",
                    f"Untrusted origin received permission: {denied_origin}",
                    "HIGH",
                )

        except requests.RequestException as error:

            self.record(
                "CORS validation",
                "FAIL",
                str(error),
                "HIGH",
            )

    # --------------------------------------------------------
    # 6. SECURITY RESPONSE HEADERS
    # --------------------------------------------------------

    def test_security_headers(self):

        try:

            response = self.request(
                "GET",
                "/",
            )

            required_headers = {
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": None,
                "Referrer-Policy": None,
                "Content-Security-Policy": None,
            }

            missing = []

            for header, expected in required_headers.items():

                value = response.headers.get(header)

                if value is None:

                    missing.append(header)

                elif expected and value.lower() != expected:

                    missing.append(
                        f"{header} (unexpected value)"
                    )

            if not missing:

                self.record(
                    "Security response headers",
                    "PASS",
                    "Checked security headers are present.",
                )

            else:

                self.record(
                    "Security response headers",
                    "WARN",
                    "Missing headers: " + ", ".join(missing),
                    "MEDIUM",
                )

        except requests.RequestException as error:

            self.record(
                "Security response headers",
                "FAIL",
                str(error),
                "HIGH",
            )

    # --------------------------------------------------------
    # 7. OPENAPI DOCUMENTATION EXPOSURE
    # --------------------------------------------------------

    def test_api_documentation(self):

        try:

            response = self.request(
                "GET",
                "/openapi.json",
            )

            if response.status_code == 200:

                self.record(
                    "OpenAPI exposure",
                    "WARN",
                    "OpenAPI documentation is publicly accessible. "
                    "Restrict it before production deployment.",
                    "LOW",
                )

            elif response.status_code in {401, 403, 404}:

                self.record(
                    "OpenAPI exposure",
                    "PASS",
                    "OpenAPI documentation is restricted or disabled.",
                )

            else:

                self.record(
                    "OpenAPI exposure",
                    "WARN",
                    f"Unexpected HTTP {response.status_code}.",
                    "LOW",
                )

        except requests.RequestException as error:

            self.record(
                "OpenAPI exposure",
                "FAIL",
                str(error),
                "MEDIUM",
            )

    # --------------------------------------------------------
    # 8. DOCTOR AUTHENTICATION CHECK
    # --------------------------------------------------------

    def test_doctor_authentication(self):

        try:

            response = self.request(
                "GET",
                "/api/v1/doctor/reviews/pending",
            )

            if response.status_code in {401, 403}:

                self.record(
                    "Doctor authentication",
                    "PASS",
                    "Endpoint requires authentication.",
                )

            elif response.status_code == 422:

                self.record(
                    "Doctor authentication",
                    "WARN",
                    "Endpoint requires doctor_id, but this is not "
                    "authentication. A signed token and role checks "
                    "are still required.",
                    "CRITICAL",
                )

            else:

                self.record(
                    "Doctor authentication",
                    "WARN",
                    f"Unexpected HTTP {response.status_code}; "
                    "verify authentication manually.",
                    "HIGH",
                )

        except requests.RequestException as error:

            self.record(
                "Doctor authentication",
                "FAIL",
                str(error),
                "HIGH",
            )

    # --------------------------------------------------------
    # 9. UPLOAD SIZE LIMIT CHECK
    # --------------------------------------------------------

    def test_upload_size_limit(self):

        try:

            oversized_content = b"x" * (
                15 * 1024 * 1024 + 1
            )

            response = self.request(
                "POST",
                "/api/v1/predict",
                files={
                    "file": (
                        "oversized.jpg",
                        oversized_content,
                        "image/jpeg",
                    )
                },
                data={
                    "explain": "false"
                },
            )

            if response.status_code in {
                400,
                413,
                415,
                422,
            }:

                self.record(
                    "Upload size limit",
                    "PASS",
                    f"Oversized upload rejected with HTTP {response.status_code}.",
                )

            elif response.status_code >= 500:

                self.record(
                    "Upload size limit",
                    "FAIL",
                    f"Server error HTTP {response.status_code}.",
                    "HIGH",
                )

            else:

                self.record(
                    "Upload size limit",
                    "FAIL",
                    f"Oversized upload was not rejected: HTTP {response.status_code}.",
                    "CRITICAL",
                )

        except requests.RequestException as error:

            self.record(
                "Upload size limit",
                "FAIL",
                str(error),
                "HIGH",
            )

    # --------------------------------------------------------
    # RUN TESTS
    # --------------------------------------------------------

    def run_all(self):

        print("\n========================================")
        print(" MEDICAL AI PLATFORM SECURITY TESTING")
        print("========================================\n")

        self.test_backend_health()
        self.test_doctor_health()
        self.test_invalid_upload()
        self.test_missing_upload()
        self.test_cors()
        self.test_security_headers()
        self.test_api_documentation()
        self.test_doctor_authentication()
        self.test_upload_size_limit()

        self.generate_report()

    # --------------------------------------------------------
    # GENERATE REPORT
    # --------------------------------------------------------

    def generate_report(self):

        counts = {
            "PASS": 0,
            "FAIL": 0,
            "WARN": 0,
        }

        for result in self.results:

            counts[result["status"]] += 1

        critical_findings = [
            result
            for result in self.results
            if result["severity"] == "CRITICAL"
        ]

        report = {
            "phase": 15,
            "name": "Security Testing",
            "generated_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "target": self.base_url,
            "summary": {
                "total_tests": len(self.results),
                "passed": counts["PASS"],
                "failed": counts["FAIL"],
                "warnings": counts["WARN"],
                "critical_findings": len(
                    critical_findings
                ),
            },
            "results": self.results,
            "production_ready": (
                counts["FAIL"] == 0
                and len(critical_findings) == 0
            ),
        }

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        REPORT_PATH.write_text(
            json.dumps(
                report,
                indent=4,
            ),
            encoding="utf-8",
        )

        print("\n========================================")
        print(" SECURITY TEST SUMMARY")
        print("========================================")

        print("Total tests:", len(self.results))
        print("Passed:", counts["PASS"])
        print("Failed:", counts["FAIL"])
        print("Warnings:", counts["WARN"])
        print("Critical findings:", len(critical_findings))

        print("\nReport saved:")
        print(REPORT_PATH)

        print(
            "\nProduction ready:",
            report["production_ready"],
        )


# ============================================================
# ENTRY POINT
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
    )

    args = parser.parse_args()

    parsed = urlparse(args.base_url)

    local_hosts = {
        "localhost",
        "127.0.0.1",
        "::1",
    }

    if parsed.hostname not in local_hosts:

        print(
            "ERROR: This test suite is restricted "
            "to localhost."
        )

        sys.exit(1)

    tester = SecurityTester(
        args.base_url
    )

    tester.run_all()


if __name__ == "__main__":

    main()
