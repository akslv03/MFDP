"""Streamlit-интерфейс для сегментации опухоли на МРТ."""

from __future__ import annotations
import json
import os
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from urllib.parse import unquote
import extra_streamlit_components as stx
import requests
import streamlit as st
import streamlit.components.v1 as components

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8080").rstrip("/")
BROWSER_API_BASE_URL = os.getenv("BROWSER_API_BASE_URL", "http://localhost:8080").rstrip("/")
DEFAULT_MODEL_ID = int(os.getenv("DEFAULT_ML_MODEL_ID", "1"))
PASSWORD_MIN_LENGTH = 4
EMAIL_PATTERN = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
AUTH_COOKIE_NAME = os.getenv("STREAMLIT_AUTH_COOKIE", "mri_access_token")
TASK_COOKIE_NAME = "mri_last_task_id"
CASE_COOKIE_NAME = "mri_open_case"
LOGOUT_COOKIE_NAME = "mri_logged_out"
AUTH_COOKIE_MAX_AGE = 60 * 60 * 12
TASK_COOKIE_MAX_AGE = 60 * 60 * 24 * 14
LOGOUT_COOKIE_MAX_AGE = 60 * 60 * 24 * 30
VIEW_QUERY_KEYS = ("task", "case", "score")


def api_url(path: str) -> str:
    return f"{API_BASE_URL}{path}"


def init_state() -> None:
    defaults = {
        "access_token": None,
        "user_id": None,
        "username": None,
        "email": None,
        "last_task_id": None,
        "selected_patient_id": None,
        "selected_case_score": None,
        "_cookie_bootstrapped": False,
        "_logged_out": False,
        "_view_restored": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_cookie_manager() -> stx.CookieManager:
    return stx.CookieManager(key="mri_auth_cookie_manager")


def bootstrap_cookies(cookie_manager: stx.CookieManager) -> None:
    cookie_manager.get_all()
    if not st.session_state._cookie_bootstrapped:
        st.session_state._cookie_bootstrapped = True
        time.sleep(0.25)
        st.rerun()


def _cookie_expires(max_age: int) -> datetime:
    return datetime.now() + timedelta(seconds=max_age)


def _cookie_set(
    cookie_manager: stx.CookieManager,
    name: str,
    value: str,
    *,
    expires_at: datetime,
    max_age: int,
    key: str,
) -> None:
    cookie_manager.set(
        name,
        value,
        expires_at=expires_at,
        max_age=max_age,
        same_site="lax",
        key=key,
    )


def _cookie_delete(cookie_manager: stx.CookieManager, name: str, *, key: str) -> None:
    try:
        cookie_manager.delete(name, key=key)
    except Exception:
        pass


def _expire_cookie(cookie_manager: stx.CookieManager, name: str, *, key: str) -> None:
    past = datetime.now() - timedelta(days=1)
    try:
        cookie_manager.set(
            name,
            "",
            expires_at=past,
            max_age=0,
            same_site="lax",
            key=key,
        )
    except Exception:
        pass


def set_auth_cookie(cookie_manager: stx.CookieManager, token: str) -> None:
    _cookie_delete(cookie_manager, LOGOUT_COOKIE_NAME, key="del_logout_flag")
    _expire_cookie(cookie_manager, LOGOUT_COOKIE_NAME, key="expire_logout_flag")
    _cookie_set(
        cookie_manager,
        AUTH_COOKIE_NAME,
        token,
        expires_at=_cookie_expires(AUTH_COOKIE_MAX_AGE),
        max_age=AUTH_COOKIE_MAX_AGE,
        key="set_auth_token",
    )
    time.sleep(0.25)


def clear_auth_cookie(cookie_manager: stx.CookieManager) -> None:
    _cookie_delete(cookie_manager, AUTH_COOKIE_NAME, key="del_auth_token")
    _expire_cookie(cookie_manager, AUTH_COOKIE_NAME, key="expire_auth_token")
    _cookie_delete(cookie_manager, "mri_auth_meta", key="del_legacy_meta")
    _expire_cookie(cookie_manager, "mri_auth_meta", key="expire_legacy_meta")


def set_logout_cookie(cookie_manager: stx.CookieManager) -> None:
    _cookie_set(
        cookie_manager,
        LOGOUT_COOKIE_NAME,
        "1",
        expires_at=_cookie_expires(LOGOUT_COOKIE_MAX_AGE),
        max_age=LOGOUT_COOKIE_MAX_AGE,
        key="set_logout_flag",
    )


def is_logout_cookie_set(cookie_manager: stx.CookieManager) -> bool:
    raw = cookie_manager.get(LOGOUT_COOKIE_NAME)
    if raw is None:
        return False
    return str(raw).strip() in {"1", "true", "True"}


def read_auth_cookie(cookie_manager: stx.CookieManager) -> Optional[str]:
    raw = cookie_manager.get(AUTH_COOKIE_NAME)
    if not raw:
        return None
    token = unquote(str(raw)).strip()
    if not token:
        return None
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    return token or None


def set_task_cookie(cookie_manager: stx.CookieManager, task_id: int) -> None:
    user_id = st.session_state.get("user_id")
    if not user_id:
        return
    _cookie_set(
        cookie_manager,
        TASK_COOKIE_NAME,
        f"{int(user_id)}:{int(task_id)}",
        expires_at=_cookie_expires(TASK_COOKIE_MAX_AGE),
        max_age=TASK_COOKIE_MAX_AGE,
        key="set_last_task",
    )


def clear_task_cookie(cookie_manager: stx.CookieManager) -> None:
    _cookie_delete(cookie_manager, TASK_COOKIE_NAME, key="del_last_task")
    _expire_cookie(cookie_manager, TASK_COOKIE_NAME, key="expire_last_task")


def read_task_cookie(cookie_manager: stx.CookieManager) -> Optional[int]:
    raw = cookie_manager.get(TASK_COOKIE_NAME)
    if not raw:
        return None
    text = unquote(str(raw)).strip()
    if not text:
        return None
    user_id = st.session_state.get("user_id")
    if user_id is None:
        return None
    if ":" in text:
        owner_raw, task_raw = text.split(":", 1)
        try:
            if int(owner_raw) != int(user_id):
                return None
            return int(task_raw)
        except (TypeError, ValueError):
            return None
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def clear_case_cookie(cookie_manager: stx.CookieManager) -> None:
    _cookie_delete(cookie_manager, CASE_COOKIE_NAME, key="del_open_case")
    _expire_cookie(cookie_manager, CASE_COOKIE_NAME, key="expire_open_case")


def clear_view_query_params() -> None:
    next_params = {k: v for k, v in st.query_params.items() if k not in VIEW_QUERY_KEYS}
    if dict(st.query_params) == next_params:
        return
    if hasattr(st.query_params, "from_dict"):
        st.query_params.from_dict(next_params)
    else:
        for key in VIEW_QUERY_KEYS:
            if key in st.query_params:
                del st.query_params[key]


def persist_view_query_params() -> None:
    next_params = {k: v for k, v in st.query_params.items() if k not in VIEW_QUERY_KEYS}
    if st.session_state.last_task_id is not None:
        next_params["task"] = str(int(st.session_state.last_task_id))
    current = dict(st.query_params)
    if current == next_params:
        return
    if hasattr(st.query_params, "from_dict"):
        st.query_params.from_dict(next_params)
    else:
        st.query_params.clear()
        st.query_params.update(next_params)


def apply_view_from_url() -> None:
    """Поднимается task из URL после обновления страницы."""
    task_raw = st.query_params.get("task")
    if task_raw and not st.session_state.last_task_id:
        try:
            st.session_state.last_task_id = int(str(task_raw))
        except (TypeError, ValueError):
            pass


def reset_workspace_view() -> None:
    """Чистая форма нового пациента (без смены пользователя)."""
    st.session_state.last_task_id = None
    st.session_state.selected_patient_id = None
    st.session_state.selected_case_score = None


def clear_session(cookie_manager: stx.CookieManager) -> None:
    st.session_state._logged_out = True
    st.session_state._view_restored = False
    for key in (
        "access_token",
        "user_id",
        "username",
        "email",
        "last_task_id",
        "selected_patient_id",
        "selected_case_score",
    ):
        st.session_state[key] = None
    clear_view_query_params()
    clear_auth_cookie(cookie_manager)
    clear_task_cookie(cookie_manager)
    clear_case_cookie(cookie_manager)
    set_logout_cookie(cookie_manager)
    time.sleep(0.35)


def start_new_patient(cookie_manager: stx.CookieManager) -> None:
    reset_workspace_view()
    clear_task_cookie(cookie_manager)
    clear_case_cookie(cookie_manager)
    persist_view_query_params()


def remember_task(cookie_manager: stx.CookieManager, task_id: int) -> None:
    st.session_state.last_task_id = task_id
    st.session_state.selected_patient_id = None
    st.session_state.selected_case_score = None
    set_task_cookie(cookie_manager, task_id)
    clear_case_cookie(cookie_manager)
    persist_view_query_params()


def open_historical_case(
    cookie_manager: stx.CookieManager,
    patient_id: str,
    score: Any = None,
) -> None:
    st.session_state.selected_patient_id = patient_id
    st.session_state.selected_case_score = score


def close_historical_case(cookie_manager: Optional[stx.CookieManager] = None) -> None:
    st.session_state.selected_patient_id = None
    st.session_state.selected_case_score = None
    if cookie_manager is not None:
        clear_case_cookie(cookie_manager)


def restore_session_from_cookie(cookie_manager: stx.CookieManager) -> None:
    if st.session_state.get("_logged_out"):
        return
    if is_logout_cookie_set(cookie_manager):
        clear_auth_cookie(cookie_manager)
        return
    if st.session_state.access_token and st.session_state.user_id:
        return

    token = read_auth_cookie(cookie_manager)
    if not token:
        return

    try:
        response = requests.get(
            api_url("/auth/me"),
            headers={"Authorization": f"Bearer {token}"},
            timeout=15,
        )
    except requests.RequestException:
        return

    if response.status_code != 200:
        clear_auth_cookie(cookie_manager)
        return

    me = response.json()
    st.session_state.access_token = token
    st.session_state.user_id = me.get("id")
    st.session_state.username = me.get("username")
    st.session_state.email = me.get("email")
    st.session_state._logged_out = False


def task_belongs_to_current_user(task_id: int) -> bool:
    try:
        response = requests.get(
            api_url(f"/api/predict/{task_id}"),
            headers=auth_headers(),
            timeout=30,
        )
    except requests.RequestException:
        return False
    return response.status_code == 200


def restore_workspace_view(cookie_manager: stx.CookieManager) -> None:
    if st.session_state.get("_view_restored"):
        return
    st.session_state._view_restored = True

    apply_view_from_url()
    clear_case_cookie(cookie_manager)
    st.session_state.selected_patient_id = None
    st.session_state.selected_case_score = None
    if "case" in st.query_params or "score" in st.query_params:
        persist_view_query_params()

    if not st.session_state.last_task_id:
        task_id = read_task_cookie(cookie_manager)
        if task_id:
            if task_belongs_to_current_user(task_id):
                st.session_state.last_task_id = task_id
            else:
                clear_task_cookie(cookie_manager)

    if st.session_state.last_task_id and not task_belongs_to_current_user(
        int(st.session_state.last_task_id)
    ):
        st.session_state.last_task_id = None
        clear_view_query_params()
        clear_task_cookie(cookie_manager)


def format_api_detail(detail: Any) -> str:
    if detail is None:
        return "Неизвестная ошибка"
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        parts: list[str] = []
        for item in detail:
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            msg = str(item.get("msg", "Ошибка валидации"))
            if msg.startswith("Value error, "):
                msg = msg[len("Value error, ") :]
            loc = item.get("loc") or []
            field = loc[-1] if loc else None
            if field == "password" and "at least" in msg.lower():
                parts.append(
                    f"Пароль должен содержать не менее {PASSWORD_MIN_LENGTH} символов"
                )
            elif field == "email":
                parts.append(
                    "Укажите корректный email. Пример: name@example.com"
                )
            else:
                parts.append(msg)
        return "; ".join(parts) if parts else "Ошибка валидации"
    return str(detail)


def validate_signup_fields(username: str, email: str, password: str) -> Optional[str]:
    if not username.strip():
        return "Укажите имя пользователя"
    if len(username.strip()) < 2:
        return "Имя пользователя должно содержать не менее 2 символов"
    if not EMAIL_PATTERN.match(email.strip()):
        return "Укажите корректный email. Пример: name@example.com"
    if len(password) < PASSWORD_MIN_LENGTH:
        return f"Пароль должен содержать не менее {PASSWORD_MIN_LENGTH} символов"
    return None


def validate_login_fields(email: str, password: str) -> Optional[str]:
    if not EMAIL_PATTERN.match(email.strip()):
        return "Укажите корректный email. Пример: name@example.com"
    if not password:
        return "Введите пароль"
    return None


def login(email: str, password: str, cookie_manager: stx.CookieManager) -> None:
    client_error = validate_login_fields(email, password)
    if client_error:
        raise RuntimeError(client_error)

    response = requests.post(
        api_url("/auth/token"),
        data={"username": email, "password": password},
        timeout=30,
    )
    if response.status_code != 200:
        raw = response.json().get("detail", response.text) if response.content else response.text
        detail = format_api_detail(raw)
        if response.status_code in (401, 403, 404):
            raise RuntimeError("Неверный email или пароль")
        raise RuntimeError(f"Ошибка входа: {detail}")

    payload = response.json()
    st.session_state.access_token = payload.get("access_token")
    st.session_state.user_id = payload["user_id"]
    st.session_state.username = payload["username"]
    st.session_state.email = payload["email"]
    st.session_state._logged_out = False
    st.session_state._view_restored = True
    reset_workspace_view()
    clear_view_query_params()
    clear_task_cookie(cookie_manager)
    if st.session_state.access_token:
        set_auth_cookie(cookie_manager, st.session_state.access_token)


def signup(username: str, email: str, password: str) -> None:
    client_error = validate_signup_fields(username, email, password)
    if client_error:
        raise RuntimeError(client_error)

    response = requests.post(
        api_url("/auth/signup"),
        json={"username": username, "email": email, "password": password},
        timeout=30,
    )
    if response.status_code not in (200, 201):
        raw = response.json().get("detail", response.text) if response.content else response.text
        detail = format_api_detail(raw)
        raise RuntimeError(detail)


SUPPORTED_UPLOAD_EXTS = (".zip", ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


def auth_headers() -> Dict[str, str]:
    token = st.session_state.access_token
    if not token:
        raise RuntimeError("Сначала войдите в аккаунт")
    return {"Authorization": f"Bearer {token}"}


def upload_and_segment(
    image_bytes: bytes,
    filename: str,
    patient_age: int,
    patient_gender: str,
) -> int:
    if not filename.lower().endswith(SUPPORTED_UPLOAD_EXTS):
        raise RuntimeError(
            "Загрузите ZIP со срезами МРТ или изображение (png/tif/jpg/bmp)"
        )

    files = {"image": (filename, image_bytes, "application/octet-stream")}
    data = {
        "ml_model_id": str(DEFAULT_MODEL_ID),
        "patient_age": str(patient_age),
        "patient_gender": patient_gender,
    }
    response = requests.post(
        api_url("/api/predict/upload"),
        data=data,
        files=files,
        headers=auth_headers(),
        timeout=180,
    )
    if response.status_code not in (200, 201):
        detail = response.json().get("detail", response.text) if response.content else response.text
        raise RuntimeError(format_api_detail(detail))
    return int(response.json()["task_id"])


def get_task(task_id: int) -> Dict[str, Any]:
    response = requests.get(
        api_url(f"/api/predict/{task_id}"),
        headers=auth_headers(),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def list_my_tasks() -> List[Dict[str, Any]]:
    response = requests.get(
        api_url("/api/predict/tasks/mine"),
        headers=auth_headers(),
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    return data if isinstance(data, list) else []


def get_historical_case(patient_id: str) -> Dict[str, Any]:
    response = requests.get(
        api_url(f"/api/cases/{patient_id}"),
        headers=auth_headers(),
        timeout=30,
    )
    if response.status_code == 404:
        raise RuntimeError("Исторический случай не найден на диске")
    response.raise_for_status()
    return response.json()


def fetch_case_slice(
    patient_id: str,
    filename: str,
    *,
    with_mask: bool = False,
) -> Optional[bytes]:
    params = {"with_mask": "true" if with_mask else "false"}
    response = requests.get(
        api_url(f"/api/cases/{patient_id}/slices/{filename}"),
        headers=auth_headers(),
        params=params,
        timeout=60,
    )
    if response.status_code != 200:
        return None
    return response.content


def render_live_slice_viewer(
    patient_id: str,
    slices: List[Dict[str, Any]],
    *,
    show_mask: bool,
) -> None:
    if not slices:
        return

    token = st.session_state.get("access_token") or ""
    files = [str(item.get("filename") or "") for item in slices]
    has_masks = [bool(item.get("has_mask")) for item in slices]
    metas = [
        (
            f"Файл: {item.get('filename')} · номер среза: {item.get('slice_number')} · "
            f"маска: {'есть' if item.get('has_mask') else 'нет'}"
        )
        for item in slices
    ]
    last = len(slices) - 1
    html = f"""
    <style>
      :root {{ color-scheme: light dark; }}
      #root, #slice-label {{
        color: light-dark(#31333F, #FAFAFA);
        font-family: "Source Sans Pro", sans-serif;
      }}
      #slice-meta, #plain-caption, #mask-caption, #slice-status {{
        color: light-dark(rgba(49, 51, 63, 0.78), rgba(250, 250, 250, 0.78));
      }}
    </style>
    <div id="root" style="box-sizing: border-box;">
      <div style="margin-bottom: 0.5rem;">
        <label id="slice-label" for="slice-slider"><b>Срез</b>:
          <span id="slice-idx">0</span> / {last}
        </label>
        <input
          id="slice-slider"
          type="range"
          min="0"
          max="{last}"
          value="0"
          style="width: 100%; margin-top: 0.35rem;"
        />
      </div>
      <div id="slice-meta" style="margin-bottom: 0.35rem; font-size: 0.95rem;"></div>
      <div id="slice-status" style="margin-bottom: 0.75rem; font-size: 0.85rem;">Загрузка среза…</div>
      <div style="display: flex; gap: 12px; align-items: flex-start;">
        <div style="flex: 1; min-width: 0;">
          <div id="plain-caption" style="font-size: 0.85rem; margin-bottom: 0.35rem;">МРТ</div>
          <img
            id="slice-plain"
            style="display: block; width: 100%; max-height: 760px; height: auto; object-fit: contain; background: #111;"
          />
        </div>
        <div id="mask-wrap" style="flex: 1; min-width: 0; display: none;">
          <div id="mask-caption" style="font-size: 0.85rem; margin-bottom: 0.35rem;">МРТ + маска</div>
          <img
            id="slice-mask"
            style="display: block; width: 100%; max-height: 760px; height: auto; object-fit: contain; background: #111;"
          />
        </div>
      </div>
    </div>
    <script>
      const apiBase = {json.dumps(BROWSER_API_BASE_URL)};
      const patientId = {json.dumps(patient_id)};
      const files = {json.dumps(files)};
      const hasMasks = {json.dumps(has_masks)};
      const metas = {json.dumps(metas)};
      const showMask = {json.dumps(bool(show_mask))};
      const token = {json.dumps(token)};
      const cache = {{ plain: {{}}, mask: {{}} }};
      const inflight = {{ plain: {{}}, mask: {{}} }};
      const slider = document.getElementById("slice-slider");
      const idxEl = document.getElementById("slice-idx");
      const metaEl = document.getElementById("slice-meta");
      const statusEl = document.getElementById("slice-status");
      const plainEl = document.getElementById("slice-plain");
      const maskEl = document.getElementById("slice-mask");
      const maskWrap = document.getElementById("mask-wrap");
      let reqSeq = 0;

      function syncColorScheme() {{
        let scheme = "dark";
        try {{
          const parentHtml = window.parent.document.documentElement;
          const parentBody = window.parent.document.body;
          const app = window.parent.document.querySelector('[data-testid="stApp"]');
          const attr =
            parentHtml.getAttribute("data-theme") ||
            parentBody.getAttribute("data-theme") ||
            parentHtml.getAttribute("data-bs-theme");
          if (attr === "light" || attr === "dark") {{
            scheme = attr;
          }} else if (app) {{
            const bg = window.parent.getComputedStyle(app).backgroundColor || "";
            const m = bg.match(/rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)/);
            if (m) {{
              const lum = (0.2126 * Number(m[1]) + 0.7152 * Number(m[2]) + 0.0722 * Number(m[3])) / 255;
              scheme = lum < 0.5 ? "dark" : "light";
            }}
          }}
          const parentScheme = window.parent.getComputedStyle(parentHtml).colorScheme;
          if (parentScheme === "light" || parentScheme === "dark") {{
            scheme = parentScheme;
          }}
        }} catch (err) {{
          scheme = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
        }}
        document.documentElement.style.colorScheme = scheme;
        document.body.style.colorScheme = scheme;
      }}

      function sliceUrl(i, withMask) {{
        const q = withMask ? "true" : "false";
        return (
          apiBase +
          "/api/cases/" + encodeURIComponent(patientId) +
          "/slices/" + encodeURIComponent(files[i]) +
          "?with_mask=" + q
        );
      }}

      function fetchSlice(i, withMask) {{
        const bucket = withMask ? "mask" : "plain";
        if (cache[bucket][i]) return Promise.resolve(cache[bucket][i]);
        if (inflight[bucket][i]) return inflight[bucket][i];
        const p = fetch(sliceUrl(i, withMask), {{
          headers: {{ Authorization: "Bearer " + token }}
        }}).then(function (r) {{
          if (!r.ok) throw new Error("HTTP " + r.status);
          return r.blob();
        }}).then(function (blob) {{
          const url = URL.createObjectURL(blob);
          cache[bucket][i] = url;
          delete inflight[bucket][i];
          return url;
        }}).catch(function (err) {{
          delete inflight[bucket][i];
          throw err;
        }});
        inflight[bucket][i] = p;
        return p;
      }}

      function prefetchAround(i) {{
        [i - 1, i + 1, i + 2].forEach(function (j) {{
          if (j < 0 || j >= files.length) return;
          fetchSlice(j, false).catch(function () {{}});
          if (showMask && hasMasks[j]) fetchSlice(j, true).catch(function () {{}});
        }});
      }}

      async function updateSlice(i) {{
        const seq = ++reqSeq;
        idxEl.textContent = String(i);
        metaEl.textContent = metas[i] || "";
        statusEl.textContent = "Загрузка среза…";
        try {{
          const plainUrl = await fetchSlice(i, false);
          if (seq !== reqSeq) return;
          plainEl.src = plainUrl;
          if (showMask && hasMasks[i]) {{
            const maskUrl = await fetchSlice(i, true);
            if (seq !== reqSeq) return;
            maskEl.src = maskUrl;
            maskWrap.style.display = "block";
          }} else {{
            maskWrap.style.display = "none";
            maskEl.removeAttribute("src");
          }}
          statusEl.textContent = "";
          prefetchAround(i);
        }} catch (err) {{
          if (seq !== reqSeq) return;
          statusEl.textContent = "Не удалось загрузить срез: " + (err && err.message ? err.message : err);
        }}
      }}

      slider.addEventListener("input", function () {{
        updateSlice(Number(slider.value));
      }});
      syncColorScheme();
      updateSlice(Number(slider.value));
      setTimeout(syncColorScheme, 50);
    </script>
    """
    components.html(html, height=860, scrolling=False)


def fetch_upload_image(path_or_name: Optional[str]) -> Optional[bytes]:
    if not path_or_name:
        return None
    filename = os.path.basename(path_or_name)
    response = requests.get(
        api_url(f"/uploads/{filename}"),
        headers=auth_headers(),
        timeout=30,
    )
    if response.status_code != 200:
        return None
    return response.content


def pick_case_preview_filename(patient_id: str, preferred: Optional[str] = None) -> Optional[str]:
    try:
        detail = get_historical_case(patient_id)
    except Exception:
        return preferred
    slices = detail.get("slices") or []
    if preferred:
        for item in slices:
            if item.get("filename") == preferred:
                return preferred
    with_mask = [item for item in slices if item.get("has_mask")]
    pool = with_mask or slices
    if not pool:
        return None
    return pool[len(pool) // 2].get("filename")


def parse_json_list(raw: Any) -> list:
    """Унифицированный парсер JSON-списка из строки или уже списка."""
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except (TypeError, json.JSONDecodeError):
        return []


def submit_doctor_review(task_id: int, decision: str) -> Dict[str, Any]:
    response = requests.post(
        api_url(f"/api/predict/{task_id}/review"),
        headers=auth_headers(),
        json={"decision": decision},
        timeout=30,
    )
    if response.status_code != 200:
        detail = response.json().get("detail", response.text) if response.content else response.text
        raise RuntimeError(format_api_detail(detail))
    return response.json()


def render_slice_triplet(display_path: Optional[str], mask_path: Optional[str], overlay_path: Optional[str]) -> None:
    cols = st.columns(3)
    original = fetch_upload_image(display_path)
    mask = fetch_upload_image(mask_path)
    overlay = fetch_upload_image(overlay_path)
    if original:
        cols[0].image(original, caption="Оригинал", use_container_width=True)
    if mask:
        cols[1].image(mask, caption="Маска опухоли", use_container_width=True)
    if overlay:
        cols[2].image(overlay, caption="Наложение", use_container_width=True)


def format_death(death: Any) -> str:
    if death == 1:
        return "умер"
    if death == 0:
        return "выжил"
    return "Нет данных"


def format_age(age: Any) -> str:
    return "Неизвестно" if age in (-1, None, "") else str(age)


def render_clinical_block(clinical: Dict[str, Any], *, score: Any = None) -> None:
    def _v(key: str, default: str = "unknown") -> Any:
        value = clinical.get(key, default)
        return default if value in (None, "") else value

    age_str = format_age(clinical.get("age"))
    st.write(f"Пациент: возраст {age_str} лет, пол (gender): {_v('gender')}")
    st.write(
        f"Опухоль: histological_type `{_v('histological_type')}`, "
        f"grade: {_v('grade')}"
    )
    st.write(
        f"Локация: {_v('location')} · laterality: {_v('laterality')} · "
        f"tumor_tissue_site: {_v('tumor_tissue_site')}"
    )
    st.write(
        f"Генетика: RNA {_v('rna_cluster')}, "
        f"метилирование {_v('meth_cluster')}, "
        f"miRNA {_v('mirna_cluster')}, "
        f"CN {_v('cn_cluster')}"
    )
    st.write(
        f"Кластеры: RPPA {_v('rppa_cluster')}, "
        f"Oncosign {_v('oncosign_cluster')}, "
        f"COC {_v('coc_cluster')}"
    )
    st.write(
        f"Раса/этничность: {_v('race')} / {_v('ethnicity')} · "
        f"исход: {format_death(clinical.get('death01'))}"
    )
    if score is not None:
        st.write(f"Визуальное сходство: `{score}`")


def render_auth(cookie_manager: stx.CookieManager) -> None:
    st.title("MRI Tumor Segmentation")

    tab_login, tab_signup = st.tabs(["Вход", "Регистрация"])

    with tab_login:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Пароль", type="password", key="login_password")
        if st.button("Войти", type="primary"):
            try:
                login(email.strip(), password, cookie_manager)
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    with tab_signup:
        username = st.text_input("Имя пользователя", key="signup_username")
        email = st.text_input("Email", key="signup_email", placeholder="name@example.com")
        password = st.text_input(
            f"Пароль (мин. {PASSWORD_MIN_LENGTH} символа)",
            type="password",
            key="signup_password",
        )
        if st.button("Зарегистрироваться"):
            try:
                signup(username.strip(), email.strip(), password)
                st.success("Аккаунт создан. Теперь войдите.")
            except Exception as exc:
                st.error(str(exc))

    st.info("Демо: `demo@client.com` / `demo_password`")


def render_history_sidebar(cookie_manager: stx.CookieManager) -> None:
    st.sidebar.subheader("История запросов")
    try:
        tasks = list_my_tasks()
    except Exception as exc:
        st.sidebar.caption(f"Не удалось загрузить историю: {exc}")
        return

    if not tasks:
        st.sidebar.caption("Пока нет запросов")
        return

    for task in tasks[:20]:
        task_id = task.get("id")
        status = task.get("status", "?")
        created = str(task.get("created_at") or "")[:19].replace("T", " ")
        age = task.get("patient_age")
        gender = task.get("patient_gender") or "—"
        label = f"№{task_id} · {status} · {created}"
        review = task.get("doctor_review") or "pending"
        if review == "accepted":
            label += " · ✓"
        elif review == "rejected":
            label += " · ✗"
        if st.sidebar.button(label, key=f"hist_{task_id}", use_container_width=True):
            remember_task(cookie_manager, int(task_id))
            st.rerun()
        st.sidebar.caption(f"Возраст {age if age is not None else '—'}, пол {gender}")


def render_case_detail(cookie_manager: stx.CookieManager) -> None:
    patient_id = st.session_state.selected_patient_id
    if not patient_id:
        return

    if st.button("← Назад", key="back_from_case"):
        close_historical_case(cookie_manager)
        st.rerun()

    st.title(f"Исторический случай `{patient_id}`")
    try:
        detail = get_historical_case(patient_id)
    except Exception as exc:
        st.error(str(exc))
        return

    clinical = detail.get("clinical") or {}
    slices = detail.get("slices") or []
    render_clinical_block(clinical, score=st.session_state.selected_case_score)
    st.caption(f"Срезов в исследовании: {detail.get('total_slices', len(slices))}")

    if not slices:
        st.warning("У пациента нет доступных срезов.")
        return

    show_mask = st.checkbox("Показать маску опухоли", value=True, key="case_show_mask")
    render_live_slice_viewer(patient_id, slices, show_mask=show_mask)


def _dismiss_case_dialog() -> None:
    st.session_state.selected_patient_id = None
    st.session_state.selected_case_score = None


@st.dialog("Исторический случай", width="large", on_dismiss=_dismiss_case_dialog)
def show_historical_case_dialog() -> None:
    patient_id = st.session_state.get("selected_patient_id")
    if not patient_id:
        return
    st.markdown(f"**Пациент** `{patient_id}`")
    try:
        detail = get_historical_case(str(patient_id))
    except Exception as exc:
        st.error(str(exc))
        return

    clinical = detail.get("clinical") or {}
    slices = detail.get("slices") or []
    render_clinical_block(clinical, score=st.session_state.selected_case_score)
    st.caption(f"Срезов в исследовании: {detail.get('total_slices', len(slices))}")
    if not slices:
        st.warning("У пациента нет доступных срезов.")
        return
    show_mask = st.checkbox(
        "Показать маску опухоли",
        value=True,
        key="case_show_mask_dialog",
    )
    render_live_slice_viewer(str(patient_id), slices, show_mask=show_mask)


def render_upload_form(cookie_manager: stx.CookieManager) -> None:
    st.subheader("Данные пациента")

    col1, col2 = st.columns(2)
    with col1:
        patient_age = st.number_input(
            "Возраст",
            min_value=0,
            max_value=120,
            value=55,
            step=1,
        )
    with col2:
        patient_gender = st.selectbox(
            "Пол",
            options=["1", "2"],
            index=0,
        )

    st.subheader("Загрузка МРТ")
    uploaded = st.file_uploader(
        "ZIP со срезами одного пациента или одно изображение МРТ",
        type=["zip", "tif", "tiff", "png", "jpg", "jpeg", "bmp"],
    )
    if uploaded is not None:
        st.info(f"Файл: `{uploaded.name}` ({len(uploaded.getvalue()) // 1024} KB)")

    if st.button("Запустить сегментацию", type="primary", disabled=uploaded is None):
        try:
            task_id = upload_and_segment(
                image_bytes=uploaded.getvalue(),
                filename=uploaded.name,
                patient_age=int(patient_age),
                patient_gender=patient_gender,
            )
            remember_task(cookie_manager, task_id)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))


def render_task_result(cookie_manager: stx.CookieManager, task_id: int) -> None:
    st.subheader("Результат")
    status_box = st.empty()

    terminal = {"completed", "failed"}
    task = None
    for _ in range(180):
        try:
            task = get_task(task_id)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                start_new_patient(cookie_manager)
                st.rerun()
                return
            status_box.error(f"Ошибка получения статуса: {exc}")
            return
        except Exception as exc:
            status_box.error(f"Ошибка получения статуса: {exc}")
            return

        status = task.get("status")
        status_box.markdown(f"Статус: `{status}` · задача `№{task_id}`")
        if status in terminal:
            break
        time.sleep(2)
    else:
        status_box.warning("Сегментация выполняется. Обновите страницу чуть позже.")
        return

    if task.get("status") == "failed":
        status_box.error(task.get("error_message") or "Сегментация не удалась")
        return

    meta_bits = []
    if task.get("patient_age") is not None:
        meta_bits.append(f"возраст {task['patient_age']}")
    if task.get("patient_gender"):
        meta_bits.append(f"пол {task['patient_gender']}")
    if meta_bits:
        st.caption("Запрос: " + ", ".join(meta_bits))

    gallery = parse_json_list(task.get("slice_gallery"))
    best = next((item for item in gallery if item.get("is_best")), None)

    st.markdown("**Лучший срез** (по максимальной площади опухоли)")
    if best:
        render_slice_triplet(
            best.get("display_image_path"),
            best.get("mask_path"),
            best.get("overlay_image_path"),
        )
        st.caption(f"Файл: `{best.get('name')}`")
    else:
        render_slice_triplet(
            task.get("display_image_path") or task.get("image_url"),
            task.get("result_mask_path"),
            task.get("overlay_image_path"),
        )

    other_slices = [item for item in gallery if not item.get("is_best")]
    if other_slices:
        with st.expander(f"Еще срезы ({len(other_slices)})", expanded=False):
            for item in other_slices:
                st.markdown(
                    f"Срез `{item.get('name')}` · площадь опухоли `{item.get('tumor_area', 0):.0f}`"
                )
                render_slice_triplet(
                    item.get("display_image_path"),
                    item.get("mask_path"),
                    item.get("overlay_image_path"),
                )
                st.divider()

    st.subheader("Оценка результата")
    review = task.get("doctor_review") or "pending"
    if review == "accepted":
        st.success("Результат принят врачом")
    elif review == "rejected":
        st.error("Результат отклонен врачом")
    else:
        st.caption("Проверьте сегментацию и примите решение.")
        col_ok, col_bad = st.columns(2)
        with col_ok:
            if st.button("Принять", type="primary", use_container_width=True, key=f"accept_{task_id}"):
                try:
                    submit_doctor_review(task_id, "accepted")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
        with col_bad:
            if st.button("Отклонить", use_container_width=True, key=f"reject_{task_id}"):
                try:
                    submit_doctor_review(task_id, "rejected")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    cases = parse_json_list(task.get("similarity_cases"))
    st.subheader("Похожие клинические случаи")
    if not cases:
        st.write("Похожие случаи не найдены. Проверьте, что Qdrant проиндексирован.")
        return

    render_similar_cases_fragment(cases, cookie_manager)


@st.fragment
def render_similar_cases_fragment(
    cases: List[Dict[str, Any]],
    cookie_manager: stx.CookieManager,
) -> None:
    if st.session_state.get("selected_patient_id"):
        show_historical_case_dialog()

    for i, case in enumerate(cases, start=1):
        patient_id = case.get("patient_id")
        st.markdown(
            f"**[{i}] Исторический случай** `{patient_id}` · "
            f"визуальное сходство `{case.get('score')}`"
        )
        render_clinical_block(case, score=None)
        volume = case.get("estimated_tumor_volume")
        if volume is not None:
            st.write(f"Объем маски: {float(volume):.1f} пикселей")

        img_cols = st.columns(2)
        preview_name = pick_case_preview_filename(
            patient_id,
            preferred=case.get("best_slice_filename"),
        )
        if preview_name:
            preview = fetch_case_slice(patient_id, preview_name, with_mask=False)
            overlay_case = fetch_case_slice(patient_id, preview_name, with_mask=True)
            if preview:
                img_cols[0].image(
                    preview,
                    caption="Исторический МРТ",
                    use_container_width=True,
                )
            if overlay_case:
                img_cols[1].image(
                    overlay_case,
                    caption="Локализация опухоли",
                    use_container_width=True,
                )

        if patient_id and st.button(
            "Подробнее",
            key=f"open_case_{i}_{patient_id}",
        ):
            open_historical_case(cookie_manager, patient_id, case.get("score"))
            show_historical_case_dialog()
        st.divider()


def render_workspace(cookie_manager: stx.CookieManager) -> None:
    st.title("Сегментация опухоли на МРТ")
    st.caption(f"Пользователь: **{st.session_state.username}** · {st.session_state.email}")

    top_cols = st.columns([1, 1, 4])
    with top_cols[0]:
        if st.button("Новый пациент", use_container_width=True):
            start_new_patient(cookie_manager)
            st.rerun()
    with top_cols[1]:
        if st.button("Выйти", use_container_width=True):
            clear_session(cookie_manager)
            st.rerun()

    if not st.session_state.access_token:
        return

    render_history_sidebar(cookie_manager)

    if st.session_state.last_task_id:
        render_task_result(cookie_manager, int(st.session_state.last_task_id))
    elif st.session_state.selected_patient_id:
        render_case_detail(cookie_manager)
    else:
        render_upload_form(cookie_manager)


def main() -> None:
    st.set_page_config(
        page_title="MRI Tumor Segmentation",
        layout="wide",
    )
    init_state()
    cookie_manager = get_cookie_manager()
    bootstrap_cookies(cookie_manager)
    restore_session_from_cookie(cookie_manager)
    if st.session_state.access_token and st.session_state.user_id:
        restore_workspace_view(cookie_manager)

    st.sidebar.title("MRI Segmentation")

    if st.session_state.access_token and st.session_state.user_id:
        render_workspace(cookie_manager)
    else:
        render_auth(cookie_manager)


if __name__ == "__main__":
    main()
