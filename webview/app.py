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

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8080").rstrip("/")
DEFAULT_MODEL_ID = int(os.getenv("DEFAULT_ML_MODEL_ID", "1"))
PASSWORD_MIN_LENGTH = 4
EMAIL_PATTERN = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
AUTH_COOKIE_NAME = os.getenv("STREAMLIT_AUTH_COOKIE", "mri_access_token")
AUTH_META_COOKIE = "mri_auth_meta"
TASK_COOKIE_NAME = "mri_last_task_id"
AUTH_COOKIE_MAX_AGE = 3600
TASK_COOKIE_MAX_AGE = 60 * 60 * 24 * 14


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
        "_task_cookie_restored": False,
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
        time.sleep(0.15)
        st.rerun()


def _cookie_expires(max_age: int) -> datetime:
    return datetime.now() + timedelta(seconds=max_age)


def set_auth_cookie(cookie_manager: stx.CookieManager, token: str) -> None:
    expires = _cookie_expires(AUTH_COOKIE_MAX_AGE)
    cookie_manager.set(
        AUTH_COOKIE_NAME,
        token,
        expires_at=expires,
        max_age=AUTH_COOKIE_MAX_AGE,
        same_site="lax",
        key="set_auth_token",
    )
    meta = {
        "user_id": st.session_state.user_id,
        "username": st.session_state.username,
        "email": st.session_state.email,
    }
    cookie_manager.set(
        AUTH_META_COOKIE,
        json.dumps(meta, ensure_ascii=False),
        expires_at=expires,
        max_age=AUTH_COOKIE_MAX_AGE,
        same_site="lax",
        key="set_auth_meta",
    )


def clear_auth_cookie(cookie_manager: stx.CookieManager) -> None:
    cookie_manager.delete(AUTH_COOKIE_NAME, key="del_auth_token")
    cookie_manager.delete(AUTH_META_COOKIE, key="del_auth_meta")
    past = datetime.now() - timedelta(days=1)
    cookie_manager.set(
        AUTH_COOKIE_NAME,
        "",
        expires_at=past,
        max_age=0,
        same_site="lax",
        key="expire_auth_token",
    )
    cookie_manager.set(
        AUTH_META_COOKIE,
        "",
        expires_at=past,
        max_age=0,
        same_site="lax",
        key="expire_auth_meta",
    )


def set_task_cookie(cookie_manager: stx.CookieManager, task_id: int) -> None:
    cookie_manager.set(
        TASK_COOKIE_NAME,
        str(task_id),
        expires_at=_cookie_expires(TASK_COOKIE_MAX_AGE),
        max_age=TASK_COOKIE_MAX_AGE,
        same_site="lax",
        key="set_last_task",
    )


def clear_task_cookie(cookie_manager: stx.CookieManager) -> None:
    cookie_manager.delete(TASK_COOKIE_NAME, key="del_last_task")
    past = datetime.now() - timedelta(days=1)
    cookie_manager.set(
        TASK_COOKIE_NAME,
        "",
        expires_at=past,
        max_age=0,
        same_site="lax",
        key="expire_last_task",
    )


def read_auth_cookie(cookie_manager: stx.CookieManager) -> Optional[str]:
    raw = cookie_manager.get(AUTH_COOKIE_NAME)
    if not raw:
        return None
    token = unquote(str(raw)).strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    return token or None


def read_task_cookie(cookie_manager: stx.CookieManager) -> Optional[int]:
    raw = cookie_manager.get(TASK_COOKIE_NAME)
    if not raw:
        return None
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return None


def clear_session(cookie_manager: stx.CookieManager) -> None:
    st.session_state._logged_out = True
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
    clear_auth_cookie(cookie_manager)
    clear_task_cookie(cookie_manager)


def start_new_patient(cookie_manager: stx.CookieManager) -> None:
    st.session_state.last_task_id = None
    st.session_state.selected_patient_id = None
    st.session_state.selected_case_score = None
    clear_task_cookie(cookie_manager)


def remember_task(cookie_manager: stx.CookieManager, task_id: int) -> None:
    st.session_state.last_task_id = task_id
    st.session_state.selected_patient_id = None
    st.session_state.selected_case_score = None
    set_task_cookie(cookie_manager, task_id)


def restore_session_from_cookie(cookie_manager: stx.CookieManager) -> None:
    if st.session_state.get("_logged_out"):
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


def restore_task_from_cookie(cookie_manager: stx.CookieManager) -> None:
    if st.session_state.get("_task_cookie_restored"):
        return
    st.session_state._task_cookie_restored = True
    if st.session_state.last_task_id:
        return
    task_id = read_task_cookie(cookie_manager)
    if task_id:
        st.session_state.last_task_id = task_id


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


def parse_slice_gallery(raw: Any) -> list:
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except (TypeError, json.JSONDecodeError):
        return []


def parse_similar_cases(raw: Any) -> list:
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
                st.success("Вход выполнен")
                time.sleep(0.2)
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


def render_case_detail() -> None:
    patient_id = st.session_state.selected_patient_id
    if not patient_id:
        return

    if st.button("← Назад", key="back_from_case"):
        st.session_state.selected_patient_id = None
        st.session_state.selected_case_score = None
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

    idx = st.slider("Срез", min_value=0, max_value=len(slices) - 1, value=0, key="case_slice_slider")
    show_mask = st.checkbox("Показать маску опухоли", value=True, key="case_show_mask")
    current = slices[idx]
    filename = current["filename"]
    st.write(
        f"Файл: `{filename}` · номер среза: {current.get('slice_number')} · "
        f"маска: {'есть' if current.get('has_mask') else 'нет'}"
    )

    cols = st.columns(2)
    plain = fetch_case_slice(patient_id, filename, with_mask=False)
    if plain:
        cols[0].image(plain, caption="МРТ", use_container_width=True)
    if show_mask and current.get("has_mask"):
        masked = fetch_case_slice(patient_id, filename, with_mask=True)
        if masked:
            cols[1].image(masked, caption="МРТ + маска", use_container_width=True)


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
        "ZIP со срезами одного пациента (рекомендуется) или одно изображение МРТ",
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


def render_task_result(task_id: int) -> None:
    st.subheader("Результат")
    status_box = st.empty()

    terminal = {"completed", "failed"}
    task = None
    for _ in range(180):
        try:
            task = get_task(task_id)
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

    gallery = parse_slice_gallery(task.get("slice_gallery"))
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

    cases = parse_similar_cases(task.get("similarity_cases"))
    st.subheader("Похожие клинические случаи")
    if not cases:
        st.write("Похожие случаи не найдены. Проверьте, что Qdrant проиндексирован.")
        return

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
            st.session_state.selected_patient_id = patient_id
            st.session_state.selected_case_score = case.get("score")
            st.rerun()
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
            time.sleep(0.2)
            st.rerun()

    render_history_sidebar(cookie_manager)

    if st.session_state.selected_patient_id:
        render_case_detail()
        return

    task_id = st.session_state.last_task_id
    if not task_id:
        render_upload_form(cookie_manager)
        return

    render_task_result(int(task_id))


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
        restore_task_from_cookie(cookie_manager)

    st.sidebar.title("MRI Segmentation")

    if st.session_state.access_token and st.session_state.user_id:
        render_workspace(cookie_manager)
    else:
        render_auth(cookie_manager)


if __name__ == "__main__":
    main()
