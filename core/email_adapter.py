"""SMTP email adapter for Site Admin alerts. No-ops when credentials are missing."""

from __future__ import annotations

import html
import os
import smtplib
import ssl
import time
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional


def _smtp_server() -> str:
    return os.environ.get("SMTP_SERVER", "smtp.gmail.com").strip() or "smtp.gmail.com"


def _smtp_port() -> int:
    try:
        return int(os.environ.get("SMTP_PORT", "465") or "465")
    except (TypeError, ValueError):
        return 465


def _username() -> str:
    return (
        os.environ.get("USERNAME_EMAIL", "").strip()
        or os.environ.get("EMAIL_FROM", "").strip()
    )


def _password() -> str:
    return os.environ.get("PASSWORD_EMAIL", "").strip()


def _from_addr() -> str:
    return os.environ.get("EMAIL_FROM", "").strip() or _username()


def configured() -> bool:
    return bool(_smtp_server() and _username() and _password() and _from_addr())


def resolve_storage_path(url_or_path: str) -> Optional[str]:
    """Map /storage/... URL to a local file path; reject path traversal."""
    raw = (url_or_path or "").strip()
    if not raw or ".." in raw.replace("\\", "/"):
        return None
    path = raw.lstrip("/").replace("\\", "/")
    if not path.startswith("storage/"):
        # Allow absolute paths under storage/
        if os.path.isabs(raw) and os.path.isfile(raw):
            return raw
        return None
    if os.path.isfile(path):
        return path
    return None


def send_email(
    to_email: str,
    subject: str,
    body: str,
    *,
    html_body: Optional[str] = None,
    inline_images: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Send plain (+ optional HTML) email via SMTP_SSL. Supports inline CID images."""
    to_email = (to_email or "").strip()
    if not configured():
        return {"ok": False, "skipped": True, "reason": "SMTP env not configured"}
    if not to_email or "@" not in to_email:
        return {"ok": False, "skipped": True, "reason": "No valid recipient email"}

    sender = _from_addr()
    text = (body or "").strip() or "(no message)"
    images = [img for img in (inline_images or []) if img.get("path") and img.get("cid")]

    if html_body and images:
        root = MIMEMultipart("related")
        alt = MIMEMultipart("alternative")
        root.attach(alt)
        alt.attach(MIMEText(text, "plain", "utf-8"))
        alt.attach(MIMEText(html_body, "html", "utf-8"))
        for img in images:
            path = img["path"]
            try:
                with open(path, "rb") as f:
                    data = f.read()
            except OSError:
                continue
            subtype = "jpeg"
            mime = (img.get("mime") or "").lower()
            if "png" in mime or path.lower().endswith(".png"):
                subtype = "png"
            elif "gif" in mime or path.lower().endswith(".gif"):
                subtype = "gif"
            part = MIMEImage(data, _subtype=subtype)
            cid = img["cid"].strip("<>")
            part.add_header("Content-ID", f"<{cid}>")
            part.add_header("Content-Disposition", "inline", filename=os.path.basename(path))
            root.attach(part)
        message = root
    else:
        message = MIMEMultipart("alternative")
        message.attach(MIMEText(text, "plain", "utf-8"))
        if html_body:
            message.attach(MIMEText(html_body, "html", "utf-8"))

    message["Subject"] = (subject or "Camera Intelligence alert")[:200]
    message["From"] = f"Camera Intelligence <{sender}>"
    message["To"] = to_email

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(_smtp_server(), _smtp_port(), context=context) as server:
            server.login(_username(), _password())
            server.sendmail(sender, to_email, message.as_string())
        return {"ok": True, "skipped": False, "to": to_email}
    except smtplib.SMTPAuthenticationError as e:
        return {"ok": False, "skipped": False, "reason": f"SMTP auth failed: {e}"}
    except Exception as e:
        return {"ok": False, "skipped": False, "reason": str(e)}


def notify_emails(
    emails: Optional[List[str]],
    subject: str,
    body: str,
    *,
    html_body: Optional[str] = None,
    inline_images: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for addr in emails or []:
        addr = str(addr).strip()
        if addr:
            results.append(
                send_email(
                    addr,
                    subject,
                    body,
                    html_body=html_body,
                    inline_images=inline_images,
                )
            )
    if not results:
        fallback = os.environ.get("EMAIL_DEFAULT_TO", "").strip()
        if fallback:
            results.append(
                send_email(
                    fallback,
                    subject,
                    body,
                    html_body=html_body,
                    inline_images=inline_images,
                )
            )
        else:
            results.append(
                {"ok": False, "skipped": True, "reason": "No recipient emails configured"}
            )
    return results


def build_alert_email(
    alert: Dict[str, Any],
    *,
    site_name: str = "",
) -> Dict[str, Any]:
    """
    Build subject/text/html + optional inline thumb for an alert.
    Returns {subject, text, html, inline_images}.
    """
    cam = str(alert.get("camera_name") or alert.get("camera_id") or "Camera").strip()
    scan = str(alert.get("scan_type") or alert.get("type") or "event").strip()
    rule_name = str(alert.get("rule_name") or "").strip() or scan.replace("_", " ")
    severity = str(alert.get("severity") or "high").strip().upper()
    msg = str(alert.get("message") or "Camera Intelligence alert").strip()
    site = (site_name or "").strip() or "your site"
    created = alert.get("created_at")
    when = ""
    if created:
        try:
            when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(created)))
        except (TypeError, ValueError):
            when = ""

    subject = f"[Alert] {rule_name} — {cam}"

    thumb_path = resolve_storage_path(str(alert.get("thumb_url") or ""))
    inline_images: List[Dict[str, str]] = []
    has_thumb = False
    if thumb_path:
        inline_images.append({"cid": "alert-thumb", "path": thumb_path, "mime": "image/jpeg"})
        has_thumb = True

    clip = str(alert.get("clip_url") or "").strip()

    text_lines = [
        "Camera Intelligence — Alert",
        f"Site: {site}",
        f"Rule broken: {rule_name}",
        f"Camera: {cam}",
        f"Type: {scan}",
        f"Severity: {severity}",
        f"Message: {msg}",
    ]
    if when:
        text_lines.append(f"Time: {when}")
    if has_thumb:
        text_lines.append("Snapshot: attached (ROI outlined)")
    if clip:
        text_lines.append(f"Clip: {clip}")
    text_lines.append("")
    text_lines.append("— Camera Intelligence")
    text = "\n".join(text_lines)

    esc = html.escape
    severity_bg = "#b45309" if severity.lower() in ("high", "critical", "danger") else "#334155"
    thumb_block = (
        f"""
        <tr>
          <td style="padding:16px 24px 8px 24px;">
            <img src="cid:alert-thumb" alt="Alert snapshot with rule ROI"
                 style="display:block;width:100%;max-width:520px;height:auto;border-radius:6px;border:1px solid #e2e8f0;" />
            <div style="font-size:12px;color:#64748b;margin-top:8px;">Snapshot with breached rule zone outlined</div>
          </td>
        </tr>
        """
        if has_thumb
        else """
        <tr>
          <td style="padding:12px 24px;color:#94a3b8;font-size:13px;">No snapshot available</td>
        </tr>
        """
    )
    clip_block = (
        f"""
        <tr>
          <td style="padding:0 24px 16px 24px;font-size:13px;color:#64748b;">
            Clip path on server: <code style="font-size:12px;">{esc(clip)}</code>
          </td>
        </tr>
        """
        if clip
        else ""
    )
    when_row = (
        f'<tr><td style="padding:4px 24px;font-size:13px;color:#64748b;">Time: {esc(when)}</td></tr>'
        if when
        else ""
    )

    html_body = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#f1f5f9;font-family:Segoe UI,Roboto,Helvetica,Arial,sans-serif;">
  <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background:#f1f5f9;padding:24px 12px;">
    <tr>
      <td align="center">
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0"
               style="max-width:560px;background:#ffffff;border-radius:10px;overflow:hidden;border:1px solid #e2e8f0;">
          <tr>
            <td style="background:#0f172a;padding:18px 24px;">
              <div style="color:#f8fafc;font-size:18px;font-weight:600;letter-spacing:0.02em;">Camera Intelligence</div>
              <div style="color:#94a3b8;font-size:12px;margin-top:4px;">{esc(site)}</div>
            </td>
          </tr>
          <tr>
            <td style="padding:20px 24px 8px 24px;">
              <span style="display:inline-block;background:{severity_bg};color:#fff;font-size:11px;font-weight:600;
                           letter-spacing:0.06em;padding:4px 10px;border-radius:4px;">{esc(severity)}</span>
              <span style="display:inline-block;margin-left:8px;background:#e2e8f0;color:#334155;font-size:11px;font-weight:600;
                           letter-spacing:0.04em;padding:4px 10px;border-radius:4px;">{esc(scan.replace("_", " ").upper())}</span>
            </td>
          </tr>
          <tr>
            <td style="padding:8px 24px 0 24px;font-size:12px;font-weight:600;letter-spacing:0.04em;color:#b45309;text-transform:uppercase;">
              Rule broken
            </td>
          </tr>
          <tr>
            <td style="padding:4px 24px 0 24px;font-size:22px;font-weight:600;color:#0f172a;">{esc(rule_name)}</td>
          </tr>
          <tr>
            <td style="padding:8px 24px 0 24px;font-size:14px;color:#64748b;">Camera: {esc(cam)}</td>
          </tr>
          <tr>
            <td style="padding:10px 24px 4px 24px;font-size:15px;line-height:1.5;color:#334155;">{esc(msg)}</td>
          </tr>
          {when_row}
          {thumb_block}
          {clip_block}
          <tr>
            <td style="padding:20px 24px;border-top:1px solid #e2e8f0;font-size:12px;color:#94a3b8;">
              Automated alert from Camera Intelligence. The snapshot highlights the rule zone that was breached.
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>"""

    return {
        "subject": subject,
        "text": text,
        "html": html_body,
        "inline_images": inline_images,
    }


def send_profile_welcome(
    to_email: str,
    contact_name: str = "",
    site_name: str = "your site",
) -> Dict[str, Any]:
    """Confirmation email after Site profile contact is saved."""
    greet = (contact_name or "").strip() or "there"
    site = (site_name or "").strip() or "your site"
    subject = f"Welcome to Camera Intelligence — {site}"
    body = (
        f"Hi {greet},\n\n"
        f"Your contact email is saved for {site}.\n"
        "You will receive Camera Intelligence alert notifications at this address "
        "when a rule includes the Email channel.\n\n"
        "You can update this anytime under Site Admin → Setup wizard or Settings.\n\n"
        "— Camera Intelligence\n"
    )
    html_body = f"""
    <html><body style="margin:0;padding:0;background:#f1f5f9;font-family:Segoe UI,Roboto,Helvetica,Arial,sans-serif;">
      <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="padding:24px 12px;">
        <tr><td align="center">
          <table role="presentation" width="100%" style="max-width:560px;background:#fff;border-radius:10px;border:1px solid #e2e8f0;">
            <tr><td style="background:#0f172a;padding:18px 24px;color:#f8fafc;font-size:18px;font-weight:600;">Camera Intelligence</td></tr>
            <tr><td style="padding:24px;color:#334155;line-height:1.5;">
              <p>Hi {html.escape(greet)},</p>
              <p>Your contact email is saved for <strong>{html.escape(site)}</strong>.</p>
              <p>You will receive alert notifications at this address when a rule includes the Email channel.</p>
              <p style="color:#94a3b8;font-size:13px;">Update this anytime under Site Admin → Setup wizard or Settings.</p>
            </td></tr>
          </table>
        </td></tr>
      </table>
    </body></html>
    """
    return send_email(to_email, subject, body, html_body=html_body)


def notify_alert_emails(
    emails: Optional[List[str]],
    alert: Dict[str, Any],
    *,
    site_name: str = "",
) -> List[Dict[str, Any]]:
    """Send a professional alert email (HTML + inline thumb) to recipients."""
    built = build_alert_email(alert, site_name=site_name)
    return notify_emails(
        emails,
        built["subject"],
        built["text"],
        html_body=built["html"],
        inline_images=built["inline_images"],
    )
