"""Premium PG Assistant — Streamlit entrypoint."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from src.bookings import bookings_for_display, create_booking
from src.complaints import CATEGORIES, PRIORITIES, STATUSES, create_complaint
from src.data_loader import list_amenity_options, load_rooms
from src.privacy import ID_REFUSAL
from src.rag import (
    answer_policy_question,
    build_policy_vectorstore,
    get_groq_api_key,
    get_groq_model,
)
from src.residents import known_room_ids, load_residents, lookup_own_record
from src.rooms import filter_rooms, free_beds_only, room_ids_with_free_beds

st.set_page_config(
    page_title="Premium PG Assistant",
    page_icon=":material/home:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Trust (teal) · Comfort (mist green) · Action (amber) · Alert (soft coral)
STATUS_COLORS = {
    "Paid": ("#2F7D4F", "#D9EEDF"),
    "Pending": ("#C9852C", "#F7E8CF"),
    "Overdue": ("#C45C4A", "#F6DDD8"),
    "Open": ("#C9852C", "#F7E8CF"),
    "In Progress": ("#2F6F8F", "#D9EAF2"),
    "Closed": ("#2F7D4F", "#D9EEDF"),
    "Low": ("#2F7D4F", "#D9EEDF"),
    "Medium": ("#C9852C", "#F7E8CF"),
    "High": ("#C45C4A", "#F6DDD8"),
}


def _status_badge(label: str) -> str:
    fg, bg = STATUS_COLORS.get(label, ("#5B6B68", "#E6EBEA"))
    return (
        f'<span style="display:inline-block;padding:0.2rem 0.65rem;'
        f"border-radius:999px;font-size:0.85rem;font-weight:600;"
        f'color:{fg};background:{bg};border:1px solid {fg}33;">{label}</span>'
    )


def _init_session() -> None:
    defaults = {
        "bookings": [],
        "complaints": [],
        "chat_messages": [],
        "prefill_room_id": None,
        "resident_lookup": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_data
def _cached_rooms() -> pd.DataFrame:
    return load_rooms()


@st.cache_data
def _cached_residents() -> pd.DataFrame:
    return load_residents()


@st.cache_resource(show_spinner="Loading PG policy index…")
def _cached_vectorstore():
    return build_policy_vectorstore()


def render_sidebar() -> None:
    st.sidebar.markdown("### :material/home: Premium PG Assistant")
    st.sidebar.caption(
        "RAG-powered help for visitors and residents. "
        "Policy answers cite handbook sections."
    )
    st.sidebar.markdown(
        """
**Privacy**
- We never show one resident’s details to another
- Phone numbers are masked
- We never ask for government ID proofs or numbers
"""
    )
    with st.sidebar.expander("Color psychology", expanded=False):
        st.caption(
            "**Teal** — trust & calm for housing decisions  \n"
            "**Mist green** — comfort / home atmosphere  \n"
            "**Amber** — pending attention, not panic  \n"
            "**Forest green** — paid / resolved success  \n"
            "**Soft coral** — overdue / high priority alerts"
        )
    key_ok = bool(get_groq_api_key())
    st.sidebar.markdown(
        f"**Groq API:** {'configured' if key_ok else 'missing — set GROQ_API_KEY'}"
    )
    if key_ok:
        st.sidebar.caption(f"Model: `{get_groq_model()}`")

    st.sidebar.info(
        "Bookings, complaints, and chat history are session-only "
        "and clear when you refresh.",
        icon=":material/info:",
    )


def tab_find_room(rooms: pd.DataFrame) -> None:
    st.subheader(":material/search: Find a room")
    st.caption("Only rooms with free beds are listed — calm teal cues signal availability & trust.")

    amenities_opts = list_amenity_options(rooms)
    food_opts = sorted(rooms["food_plan"].dropna().unique().tolist())
    free = free_beds_only(rooms)
    max_rent = int(rooms["rent_inr_per_bed_month"].max())
    min_rent = int(rooms["rent_inr_per_bed_month"].min())

    c1, c2, c3 = st.columns(3)
    with c1:
        sharing_choice = st.selectbox(
            "Sharing",
            options=["Any", 2, 3, 4],
            index=0,
        )
    with c2:
        ac_choice = st.selectbox("AC", options=["Any", "Yes", "No"], index=0)
    with c3:
        food_choice = st.selectbox(
            "Food plan",
            options=["Any", *food_opts],
            index=0,
        )

    c4, c5 = st.columns(2)
    with c4:
        budget = st.slider(
            "Max budget (₹ / bed / month)",
            min_value=min_rent,
            max_value=max_rent,
            value=max_rent,
            step=100,
        )
    with c5:
        amenity_sel = st.multiselect("Amenities", options=amenities_opts)

    filtered = filter_rooms(
        rooms,
        sharing=None if sharing_choice == "Any" else int(sharing_choice),
        ac=None if ac_choice == "Any" else ac_choice,
        food_plan=None if food_choice == "Any" else food_choice,
        amenities=amenity_sel or None,
        max_budget=budget,
    )

    st.write(
        f"**{len(filtered)}** room(s) with free beds"
        f" (of {len(free)} available overall)."
    )

    if filtered.empty:
        st.warning("No rooms match these filters. Try widening budget or amenities.")
        return

    display = filtered.rename(
        columns={
            "room_id": "Room",
            "floor": "Floor",
            "sharing": "Sharing",
            "AC": "AC",
            "food_plan": "Food",
            "amenities": "Amenities",
            "rent_inr_per_bed_month": "Rent ₹/bed/mo",
            "beds_free": "Free beds",
        }
    )
    st.dataframe(display, width="stretch", hide_index=True, alt="Available rooms matching filters")

    st.markdown("#### Request a room")
    pick = st.selectbox("Select room to request", options=filtered["room_id"].tolist())
    if st.button("Continue to booking", type="primary", icon=":material/arrow_forward:"):
        st.session_state.prefill_room_id = pick
        st.success(
            f"Room **{pick}** selected. Open the **Booking request** tab to submit.",
            icon=":material/check_circle:",
        )


def tab_booking(rooms: pd.DataFrame) -> None:
    st.subheader(":material/event_available: Booking request")
    st.caption(
        "Submit a move-in request with test contact details. "
        "Status starts as Pending (amber = needs attention). Do not enter government ID numbers."
    )

    free_ids = room_ids_with_free_beds(rooms)
    if not free_ids:
        st.error("No rooms with free beds right now.")
        return

    prefill = st.session_state.get("prefill_room_id")
    default_idx = free_ids.index(prefill) if prefill in free_ids else 0

    with st.form("booking_form", clear_on_submit=True):
        room_id = st.selectbox("Room", options=free_ids, index=default_idx)
        move_in = st.date_input("Move-in date", value=date.today())
        name = st.text_input("Full name")
        contact = st.text_input(
            "Test contact (phone or email)",
            help="Phone numbers are masked in the request list. No government IDs.",
        )
        submitted = st.form_submit_button("Submit booking request", type="primary")

    if submitted:
        if not name.strip() or not contact.strip():
            st.error("Name and contact are required.")
        elif any(x in contact.lower() for x in ("aadhaar", "aadhar", "pan ", "passport")):
            st.error(ID_REFUSAL)
        else:
            booking = create_booking(
                room_id=room_id,
                move_in_date=move_in,
                name=name,
                contact=contact,
            )
            st.session_state.bookings.append(booking)
            st.session_state.prefill_room_id = None
            st.success(
                f"Request **{booking['request_id']}** created with status "
                f"**{booking['status']}**.",
                icon=":material/check_circle:",
            )

    st.markdown("#### Your session requests")
    if not st.session_state.bookings:
        st.info("No booking requests in this session yet.", icon=":material/info:")
    else:
        rows = bookings_for_display(st.session_state.bookings)
        st.dataframe(
            pd.DataFrame(rows),
            width="stretch",
            hide_index=True,
            alt="Session booking requests",
        )
        for row in rows:
            st.markdown(
                f"**{row['request_id']}** · Room {row['room_id']} · "
                f"{_status_badge(row['status'])}",
                unsafe_allow_html=True,
            )


def tab_my_rent(residents: pd.DataFrame) -> None:
    st.subheader(":material/payments: My rent")
    st.caption(
        "Look up **only your own** rent record with nickname + room. "
        "Other residents’ details are never shown."
    )

    rooms = known_room_ids(residents)
    with st.form("resident_lookup_form"):
        nickname = st.text_input("Your nickname")
        room = st.selectbox("Your room", options=["— select —", *rooms])
        submitted = st.form_submit_button("View my rent", type="primary")

    if submitted:
        if not nickname.strip() or room == "— select —":
            st.error("Enter your nickname and select your room.")
            st.session_state.resident_lookup = None
        else:
            record = lookup_own_record(residents, nickname, room)
            st.session_state.resident_lookup = record
            if record is None:
                st.warning(
                    "No matching record for that nickname and room. "
                    "Check spelling or contact the PG manager. "
                    "We will not list other residents."
                )

    record = st.session_state.get("resident_lookup")
    if record:
        status = record["paid_status"]
        fg, bg = STATUS_COLORS.get(status, ("#5B6B68", "#E6EBEA"))
        st.html(
            f"""
            <div style="padding:1.1rem 1.25rem;border-radius:0.75rem;
                        background:{bg};border:1px solid {fg}44;margin:0.5rem 0 1rem;">
              <div style="display:flex;justify-content:space-between;align-items:center;gap:1rem;">
                <div>
                  <div style="font-size:0.85rem;color:#5B6B68;">Your rent record</div>
                  <div style="font-size:1.35rem;font-weight:700;color:#1A2E2C;">
                    {record['nickname']} · Room {record['room']}
                  </div>
                </div>
                {_status_badge(status)}
              </div>
              <div style="margin-top:0.9rem;display:grid;grid-template-columns:1fr 1fr;gap:0.75rem;">
                <div><div style="color:#5B6B68;font-size:0.8rem;">Rent</div>
                     <div style="font-weight:600;">₹{record['rent_inr']}</div></div>
                <div><div style="color:#5B6B68;font-size:0.8rem;">Due date</div>
                     <div style="font-weight:600;">{record['due_date']}</div></div>
              </div>
            </div>
            """
        )
        st.info(
            "For disputes or payment proof, contact the PG manager. "
            "Never share government ID numbers in this app.",
            icon=":material/privacy_tip:",
        )

    with st.expander("Demo tip"):
        st.write(
            "Sample nicknames in the demo data include Avi (G01), Sam (G03), "
            "Dev (G05). Use the matching room ID — we never show a resident directory."
        )


def tab_policy_chat() -> None:
    st.subheader(":material/menu_book: Policy / FAQ chat")
    st.caption(
        "Answers come from the PG Resident Policy with section citations. "
        "Uncovered questions are referred to the PG manager."
    )

    if not get_groq_api_key():
        st.warning(
            "Set `GROQ_API_KEY` in the environment or `.streamlit/secrets.toml` "
            "to enable chat."
        )

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                st.caption("Sources: " + " · ".join(msg["citations"]))

    prompt = st.chat_input("Ask about quiet hours, visitors, deposit, food…")
    if not prompt:
        return

    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Checking policy…"):
            try:
                store = _cached_vectorstore()
                result = answer_policy_question(prompt, store)
            except Exception as exc:  # noqa: BLE001
                result_answer = (
                    f"Something went wrong while retrieving policy ({exc}). "
                    "Please contact the PG manager."
                )
                st.markdown(result_answer)
                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": result_answer, "citations": []}
                )
                return

            st.markdown(result.answer)
            if result.citations:
                st.caption("Sources: " + " · ".join(result.citations))
            if result.referred_to_manager:
                st.info("Referred to PG manager for uncovered details.")

    st.session_state.chat_messages.append(
        {
            "role": "assistant",
            "content": result.answer,
            "citations": result.citations,
        }
    )


def tab_complaints() -> None:
    st.subheader(":material/report: Complaint ticket")
    st.caption(
        "Raise a ticket with category, priority, and description. "
        "High priority uses soft coral — clear urgency without harsh alarm."
    )

    with st.form("complaint_form", clear_on_submit=True):
        category = st.selectbox("Category", options=CATEGORIES)
        priority = st.selectbox("Priority", options=PRIORITIES, index=1)
        description = st.text_area("Description", height=120)
        submitted = st.form_submit_button("Submit ticket", type="primary")

    if submitted:
        if not description.strip():
            st.error("Please describe the issue.")
        else:
            ticket = create_complaint(
                category=category,
                priority=priority,
                description=description,
            )
            st.session_state.complaints.append(ticket)
            st.success(
                f"Ticket **{ticket['ticket_id']}** opened with status **{ticket['status']}**.",
                icon=":material/check_circle:",
            )

    st.markdown("#### Your session tickets")
    if not st.session_state.complaints:
        st.info("No complaint tickets in this session yet.", icon=":material/info:")
        return

    for i, ticket in enumerate(st.session_state.complaints):
        with st.container(border=True):
            c1, c2, c3 = st.columns([2, 2, 2])
            c1.markdown(
                f"**{ticket['ticket_id']}** · {ticket['category']}  \n"
                f"{_status_badge(ticket['status'])}",
                unsafe_allow_html=True,
            )
            c2.markdown(
                f"Priority  \n{_status_badge(ticket['priority'])}",
                unsafe_allow_html=True,
            )
            new_status = c3.selectbox(
                "Status",
                options=STATUSES,
                index=STATUSES.index(ticket["status"]),
                key=f"status_{ticket['ticket_id']}",
            )
            if new_status != ticket["status"]:
                st.session_state.complaints[i]["status"] = new_status
                st.rerun()
            st.write(ticket["description"])


def main() -> None:
    _init_session()
    render_sidebar()

    st.html(
        """
        <div style="
            margin:0 0 1.25rem 0;padding:1.35rem 1.5rem;border-radius:0.9rem;
            background:linear-gradient(120deg,#0A4F52 0%,#0D7377 45%,#3A9B8F 100%);
            color:#F3F7F6;box-shadow:0 8px 24px rgba(10,79,82,0.18);">
          <div style="font-size:0.8rem;letter-spacing:0.06em;text-transform:uppercase;
                      opacity:0.85;margin-bottom:0.35rem;">Premium PG Assistant</div>
          <div style="font-size:1.75rem;font-weight:700;line-height:1.2;margin-bottom:0.4rem;">
            Feel at home — decide with clarity
          </div>
          <div style="opacity:0.92;max-width:42rem;font-size:1rem;">
            Teal builds trust for room seekers; soft greens keep the space calm for residents.
            Policy answers stay grounded, private, and citation-backed.
          </div>
        </div>
        """
    )

    rooms = _cached_rooms()
    residents = _cached_residents()

    tabs = st.tabs(
        [
            "Find a Room",
            "Booking Request",
            "My Rent",
            "Policy / FAQ Chat",
            "Complaints",
        ]
    )
    with tabs[0]:
        tab_find_room(rooms)
    with tabs[1]:
        tab_booking(rooms)
    with tabs[2]:
        tab_my_rent(residents)
    with tabs[3]:
        tab_policy_chat()
    with tabs[4]:
        tab_complaints()


if __name__ == "__main__":
    main()
