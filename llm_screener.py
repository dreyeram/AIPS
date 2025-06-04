import streamlit as st
import requests
import json # For parsing LLM responses
import os # For path joining if needed, though direct filenames work here

# --- Configuration ---
OPENROUTER_API_KEY_SECRET = st.secrets.get("OPENROUTER_API_KEY", "")
LLM_MODEL_SECRET = st.secrets.get("LLM_MODEL", "mistralai/mistral-7b-instruct:free")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# --- Load Prompts from Files ---
def load_prompt(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        st.error(f"Prompt file not found: {file_path}. Please ensure it's in your GitHub repository.")
        return None
    except Exception as e:
        st.error(f"Error loading prompt file {file_path}: {e}")
        return None

SYSTEM_PROMPT = load_prompt("system_prompt.txt")
SUMMARIZATION_PROMPT_TEMPLATE = load_prompt("summarization_prompt.txt")

# --- LLM Interaction Function ---
def get_llm_response(conversation_history):
    if not OPENROUTER_API_KEY_SECRET:
        st.error("OpenRouter API Key is not configured in app secrets.")
        return None
    if not SYSTEM_PROMPT or not SUMMARIZATION_PROMPT_TEMPLATE: # Check if prompts loaded
        st.error("One or more prompt files could not be loaded. Aborting LLM call.")
        return None

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY_SECRET}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://your-app-name.streamlit.app", # Optional
    }
    payload = {
        "model": LLM_MODEL_SECRET,
        "messages": conversation_history
    }
    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, data=json.dumps(payload), timeout=180)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        st.error(f"API Request Error: {e}")
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        response_text_for_error = response.text if 'response' in locals() else 'No response object'
        st.error(f"Error parsing LLM response: {e} - Response: {response_text_for_error[:500]}...") # Show part of response
    return None

# --- Streamlit App UI and Logic ---
st.set_page_config(layout="wide", page_title="Holistic Health Screener")
st.title("🌿 Holistic Health Insight Assistant 🌿")

st.sidebar.header("About this Tool")
st.sidebar.info(
    """
    This tool uses an AI to help you gather your health information.
    - Answer the AI's questions. Some may offer clickable options.
    - When the AI indicates the assessment is concluding, or if you type "END ASSESSMENT", the process will move towards summarization.
    - A summary will be generated for you to review and share with your doctor.
    **This tool does NOT provide medical advice or diagnosis.**
    """
)
st.sidebar.markdown(f"**Using Model:** `{LLM_MODEL_SECRET}` (configured in secrets)")
st.sidebar.markdown("---")
if st.sidebar.button("Clear Chat & Restart Assessment"):
    # Clear specific relevant keys instead of st.session_state.clear() to preserve others if any
    keys_to_clear = ["messages", "current_llm_question_data", "assessment_phase_complete", 
                     "summary_generated", "summary_text", "initial_greeting_done"]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_llm_question_data" not in st.session_state:
    st.session_state.current_llm_question_data = None
if "assessment_phase_complete" not in st.session_state:
    st.session_state.assessment_phase_complete = False
if "summary_generated" not in st.session_state:
    st.session_state.summary_generated = False
if "summary_text" not in st.session_state:
    st.session_state.summary_text = ""
if "initial_greeting_done" not in st.session_state: # To handle initial conversational greeting
    st.session_state.initial_greeting_done = False


# --- Helper function to process LLM response and update UI state ---
def process_llm_response(response_text):
    st.session_state.current_llm_question_data = None # Reset
    is_structured_json_question = False
    message_to_display_in_chat = response_text # Default to full response

    try:
        # Assumes LLM strictly follows "JSON only" rule for interactive questions.
        data = json.loads(response_text)
        if isinstance(data, dict) and "input_type" in data and "question_text" in data:
            st.session_state.current_llm_question_data = data
            message_to_display_in_chat = data["question_text"] # Display only the question part
            is_structured_json_question = True
            if data["input_type"] == "text_display_only":
                st.session_state.assessment_phase_complete = True
        # else: LLM returned JSON but not our expected interactive question structure. Treat as plain text.
    except json.JSONDecodeError:
        # LLM didn't return a pure JSON string. Treat as a plain conversational message.
        # This will happen for the initial greeting, for example.
        st.session_state.current_llm_question_data = {"question_text": response_text, "input_type": "text"}
        # message_to_display_in_chat is already response_text

    st.session_state.messages.append({"role": "assistant", "content": message_to_display_in_chat})


# --- Initial greeting from LLM ---
if not st.session_state.initial_greeting_done and OPENROUTER_API_KEY_SECRET and SYSTEM_PROMPT:
    if not st.session_state.messages: # Only send if messages list is truly empty
        with st.spinner("Holistic Health Insight Assistant is starting up..."):
            # For the very first call, the conversation_history only contains the system prompt.
            # The LLM is instructed to give a conversational intro first.
            initial_response_text = get_llm_response(
                conversation_history=[{"role": "system", "content": SYSTEM_PROMPT}]
            )
        if initial_response_text:
            # The first response *should be* conversational text, not JSON, as per prompt.
            process_llm_response(initial_response_text) 
            st.session_state.initial_greeting_done = True # Mark greeting as done
        else:
            st.session_state.messages.append({"role": "assistant", "content": "Hello! I'm ready to start. (Error occurred during initial greeting setup). To begin, could you please tell me your current age?"})
            st.session_state.current_llm_question_data = {"question_text": "To begin, could you please tell me your current age?", "input_type": "text"}
            st.session_state.initial_greeting_done = True # Mark as done to prevent retries
        st.rerun()


# --- Display chat messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --- Handle user input based on current LLM question type ---
if OPENROUTER_API_KEY_SECRET and SYSTEM_PROMPT and st.session_state.initial_greeting_done and \
   not st.session_state.assessment_phase_complete and not st.session_state.summary_generated:
    
    llm_q_data = st.session_state.current_llm_question_data

    if llm_q_data and llm_q_data["input_type"] == "options":
        options_data = llm_q_data.get("options", [])
        allow_multiple = llm_q_data.get("allow_multiple_selections", False)
        
        # Form key needs to be dynamic if question text changes to ensure form re-renders
        form_key = f"options_form_{hash(llm_q_data['question_text'])}"

        with st.form(key=form_key):
            # We don't display llm_q_data["question_text"] here again because it's already in chat.
            # st.markdown(llm_q_data["question_text"]) 
            
            formatted_options = [f"{opt['value']} (e.g., {opt['example']})" if 'example' in opt and opt['example'] else opt['value'] for opt in options_data]
            
            selections_from_ui = []
            if allow_multiple:
                selections_from_ui = st.multiselect("Select all that apply:", formatted_options, key=f"ms_{form_key}")
            else:
                selection_from_ui_single = st.radio("Choose one:", formatted_options, key=f"rad_{form_key}")
                if selection_from_ui_single:
                    selections_from_ui = [selection_from_ui_single]
            
            other_option_details = next((opt for opt in options_data if "other" in opt.get("value","").lower()), None)
            other_text = ""
            if other_option_details: # Check if "Other" option exists
                other_text = st.text_input("If 'Other', please specify:", key=f"other_{form_key}")

            submit_button = st.form_submit_button(label="Submit Answer")

            if submit_button:
                user_response_parts = []
                
                for raw_sel_ui in selections_from_ui:
                    original_value = raw_sel_ui.split(" (e.g.,")[0].strip() # Get original value
                    user_response_parts.append(original_value)
                    # If this selection was "Other" and they typed something, use their text
                    if "other" in original_value.lower() and other_text:
                        user_response_parts[-1] = f"Other: {other_text.strip()}" 
                
                # If "Other" wasn't among selections but text was entered (e.g. user cleared selections but typed other)
                if not any("other" in part.lower() for part in user_response_parts) and other_option_details and other_text:
                     user_response_parts.append(f"Other: {other_text.strip()}")
                # Handle case where "Other" is selected but nothing typed.
                elif any("other" in part.lower() for part in user_response_parts) and not other_text and not any("Other:" in part for part in user_response_parts):
                    # Find the "Other" selection and mark it as specified but empty
                    for i, part in enumerate(user_response_parts):
                        if "other" in part.lower():
                            user_response_parts[i] = "Other (not specified)"
                            break


                user_response_full = "; ".join(user_response_parts) if user_response_parts else "No specific option selected (or cleared selection)."
                
                st.session_state.messages.append({"role": "user", "content": user_response_full})
                st.session_state.current_llm_question_data = None 

                with st.spinner("Thinking..."):
                    llm_response_text = get_llm_response(
                        conversation_history=[{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.messages
                    )
                if llm_response_text:
                    process_llm_response(llm_response_text)
                st.rerun()

    elif llm_q_data and llm_q_data["input_type"] == "text":
        if prompt := st.chat_input("Your answer:", key=f"text_chat_input_{hash(llm_q_data['question_text'])}"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.current_llm_question_data = None 

            if "END ASSESSMENT" in prompt.upper() and len(st.session_state.messages) > 3:
                st.session_state.assessment_phase_complete = True
                final_assistant_message_content = "Understood. The information gathering is complete. When you're ready, please click the 'Generate Doctor's Summary' button."
                st.session_state.messages.append({"role": "assistant", "content": final_assistant_message_content})
                # Ensure this last message doesn't get parsed as a question
                st.session_state.current_llm_question_data = {"question_text": final_assistant_message_content, "input_type": "text_display_only"}
                st.rerun()
            else:
                with st.spinner("Thinking..."):
                    llm_response_text = get_llm_response(
                        conversation_history=[{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.messages
                    )
                if llm_response_text:
                    process_llm_response(llm_response_text)
                st.rerun()
    
    elif not llm_q_data and st.session_state.initial_greeting_done and st.session_state.messages:
        # This state can occur if the last LLM call failed or didn't set current_llm_question_data
        # We can offer a way to retry or just wait. For now, let's indicate waiting.
        # Or, if the last message was from user, we should be expecting an LLM response.
        if st.session_state.messages[-1]["role"] == "user":
            with st.spinner("Waiting for assistant..."):
                llm_response_text = get_llm_response(
                    conversation_history=[{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.messages
                )
            if llm_response_text:
                process_llm_response(llm_response_text)
                st.rerun()
            else:
                st.warning("Having trouble getting a response from the assistant. Please check logs or try restarting.")


# --- Generate Summary Button ---
if st.session_state.assessment_phase_complete and not st.session_state.summary_generated and OPENROUTER_API_KEY_SECRET and SUMMARIZATION_PROMPT_TEMPLATE:
    if st.button("Generate Doctor's Summary", type="primary"):
        with st.spinner("Generating summary for your doctor... This may take a moment."):
            conversation_for_summary = "\n".join(
                [f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages]
            )
            summarization_full_prompt_for_llm = SUMMARIZATION_PROMPT_TEMPLATE.format( # Use .format()
                conversation_history_text=conversation_for_summary
            )
            summary_payload_messages = [
                {"role": "system", "content": "You are an AI assistant specialized in summarizing patient conversations into structured medical reports for doctors. Follow the user's instructions precisely."},
                {"role": "user", "content": summarization_full_prompt_for_llm}
            ]
            summary_text_response = get_llm_response(summary_payload_messages)

            if summary_text_response:
                st.session_state.summary_text = summary_text_response
                st.session_state.summary_generated = True
                st.success("Summary generated successfully!")
            else:
                st.error("Could not generate the summary. Please check API key and try again if needed.")
            st.rerun()

# --- Display Generated Summary ---
if st.session_state.summary_generated:
    st.markdown("---")
    st.subheader("📋 Summary for Your Doctor")
    st.markdown(
        "Please review this summary carefully. You can copy it or download it to share with your doctor. "
        "Remember, this is based on your self-reported information and is not a diagnosis."
    )
    st.text_area("Doctor's Summary:", value=st.session_state.summary_text, height=600, key="summary_display_area_final_v2")
    st.download_button(
        label="Download Summary as Text File",
        data=st.session_state.summary_text.encode("utf-8"),
        file_name="holistic_health_summary.txt",
        mime="text/plain"
    )

elif not OPENROUTER_API_KEY_SECRET:
    st.error("OpenRouter API Key is not configured in your app's secrets. Please ask the app administrator to set it up.")
elif not SYSTEM_PROMPT or not SUMMARIZATION_PROMPT_TEMPLATE:
     st.error("Core prompt files are missing. The application cannot function correctly. Please check the deployment.")
