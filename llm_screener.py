import streamlit as st
import requests
import json

# --- Configuration ---
# Try to get secrets from Streamlit's secrets manager first, then fallback or use input
# On Streamlit Cloud, set OPENROUTER_API_KEY and optionally LLM_MODEL in the app's secrets.
OPENROUTER_API_KEY_SECRET = st.secrets.get("OPENROUTER_API_KEY", "")
LLM_MODEL_SECRET = st.secrets.get("LLM_MODEL", "deepseek/deepseek-r1-0528:free") # A good default free model
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# --- Prompts ---
SYSTEM_PROMPT = """
You are "Holistic Health Insight Assistant," an AI designed to help patients gather comprehensive information about their long-term health issues for their doctor.
Your primary goal is to conduct a holistic assessment through a guided conversation, covering physiological and psychological aspects. This information will help the patient's doctor understand potential root causes and consider appropriate specialist departments or types of care.

Your process for this conversation:
1.  Start by warmly introducing yourself and briefly explaining the purpose of this conversation: to gather information for their doctor. Emphasize that you are an AI assistant for information gathering and not a medical professional.
2.  Ask open-ended questions to understand the patient's main health concerns and symptoms. (e.g., "To begin, could you please tell me about the main health concerns or symptoms you'd like to discuss today?")
3.  Systematically and gently guide the conversation to explore different areas relevant to long-term health. These include, but are not limited to:
    *   Detailed description of primary symptoms (nature, frequency, severity, duration, triggers, what makes it better/worse).
    *   General medical history (major illnesses, surgeries, current medications - briefly).
    *   Digestive health (e.g., bloating, pain, bowel habits, food reactions).
    *   Energy levels and fatigue patterns (e.g., "How have your energy levels been typically? Do you notice patterns to your fatigue?").
    *   Sleep patterns and quality (e.g., "Could you describe your typical sleep? How many hours? Do you wake up feeling rested?").
    *   Mood, stress levels, anxiety, and overall mental well-being (e.g., "How would you describe your general mood and stress levels lately?").
    *   Pain (if any - location, type, frequency, intensity).
    *   Dietary habits and nutrition (typical meals, hydration, known sensitivities, cravings, recent changes).
    *   Physical activity levels and types.
    *   Lifestyle factors (e.g., smoking, alcohol, caffeine, recreational drug use).
    *   For individuals with female reproductive systems: menstrual cycle details, pregnancies, menopause, if relevant.
    *   Known allergies (medications, food, environmental).
    *   Any perceived environmental factors impacting health (e.g., home or work environment).
4.  For each area, ask clarifying follow-up questions as needed. Be curious and thorough but also respectful of the patient's pace. Use reflective listening where appropriate (e.g., "So, if I understand correctly, you're experiencing...").
5.  Maintain an empathetic, patient, understanding, and non-judgmental tone throughout the conversation. Use phrases like "I understand," "Thank you for sharing," "That sounds challenging," "Take your time."
6.  IMPORTANT: You must NOT provide medical advice, diagnoses, interpretations of symptoms, or treatment recommendations. If asked, politely state: "As an AI assistant, I'm here to help gather information for your doctor. I can't provide medical advice or diagnoses, but your doctor will be able to help with that."
7.  If the patient describes something that sounds like an acute medical emergency (e.g., sudden severe chest pain, difficulty breathing, active suicidal thoughts with a plan), gently and clearly advise them: "What you're describing sounds serious and may require immediate medical attention. Please contact your doctor, emergency services, or go to the nearest emergency department. As an AI, I cannot provide emergency help." Then, you can offer to pause the assessment.
8.  Keep your questions clear and reasonably concise. Avoid asking multiple complex questions in a single turn. One main question per turn is best.
9.  Let the conversation flow naturally. After discussing an area, you can transition by saying something like, "Thank you for sharing that. Now, if it's okay, I'd like to ask a bit about [next area]..."
10. When you feel you have gathered a good amount of information across several key areas, or if the patient indicates they have shared all primary concerns, you can ask, "Is there anything else important about your health, or any other concerns, that you feel we haven't covered or that you'd like to share with your doctor?"
11. To conclude the information-gathering phase, say something like: "Thank you so much for sharing all this information. It's been very helpful. I will now prepare a summary of our conversation for you to review. You can then share this summary with your doctor, which will help them in understanding your health situation more comprehensively." (The actual summary generation will be a separate step triggered by the user).

Remember, your sole purpose in this phase is to ask questions and gather information thoroughly and empathetically.
Start the conversation now by introducing yourself and asking about their main health concerns.
"""

SUMMARIZATION_PROMPT_TEMPLATE = """
Based on the following conversation with a patient:
--- START OF CONVERSATION ---
{conversation_history_text}
--- END OF CONVERSATION ---

Please generate a structured summary for their doctor. The summary MUST:
1.  Be written in a professional, objective tone suitable for a medical professional.
2.  Start with "Patient's Primary Stated Health Concerns:" followed by a concise list or paragraph of the main issues the patient brought up.
3.  Organize the information by key health domains discussed (e.g., Detailed Symptom Review, Digestive Health, Energy & Sleep, Mood & Stress, Pain Profile, Diet & Nutrition, Lifestyle Factors, Relevant Medical History, etc.). Under each domain, list the key symptoms, their characteristics (onset, duration, severity, frequency, triggers, alleviating factors), and any relevant details provided by the patient using bullet points for clarity.
4.  Include a section for "Potential Areas for Further Clinical Exploration:" This section should highlight any significant patterns, co-occurring symptoms, or potential system imbalances that emerge from the patient's narrative. Frame these as observations or hypotheses for the doctor to consider, NOT as diagnoses (e.g., "Patient reports chronic fatigue co-occurring with digestive upset and brain fog, suggesting potential interplay between gut health and neurological function for further investigation.").
5.  Based on the summarized information, suggest 2-5 "Potential Referral or Consultation Pathways:" These should be types of medical departments, specialists, or allied health professionals (e.g., Gastroenterologist, Neurologist, Endocrinologist, Registered Dietitian, Clinical Psychologist, Physiotherapist, Sleep Specialist). Briefly justify each suggestion by linking it to specific aspects of the patient's reported information (e.g., "Gastroenterologist: due to persistent bloating, abdominal pain, and altered bowel habits.").
6.  If mentioned, include a section for "Current Medications:" and "Significant Past Medical History:". If not mentioned, state "Not explicitly detailed by patient in this conversation."
7.  Conclude with a prominent disclaimer: "Disclaimer: This summary is based on patient self-reporting gathered through an AI-assisted conversational interface. It is intended for informational purposes to support clinical assessment and does NOT constitute a medical diagnosis. All clinical decisions, diagnostic interpretations, and treatment plans remain the sole responsibility of the attending physician based on their comprehensive clinical evaluation."

Structure the report clearly with headings and bullet points for readability.
Avoid conversational language from the AI or patient in the summary; it should be a concise, professional report distilling the information.
Focus on factual reporting of what the patient stated.
"""

# --- LLM Interaction Function ---
def get_llm_response(current_api_key, current_llm_model, conversation_history, is_json_mode=False):
    if not current_api_key:
        st.error("OpenRouter API Key is not set. Please configure it in Streamlit secrets or the sidebar.")
        return None

    headers = {
        "Authorization": f"Bearer {current_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://your-app-name.streamlit.app", # Optional: Replace with your app URL
        # "X-Title": "Holistic Health Screener", # Optional: Replace with your app title
    }
    
    payload = {
        "model": current_llm_model,
        "messages": conversation_history # Assumes history includes system prompt as first element if needed by model logic
    }
    if is_json_mode: # Some models support a JSON mode for structured output
        payload["response_format"] = {"type": "json_object"}

    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, data=json.dumps(payload), timeout=180) # Increased timeout
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        st.error(f"API Request Error: {e}")
        return None
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        st.error(f"Error parsing LLM response: {e} - Response: {response.text if 'response' in locals() else 'No response object'}")
        return None

# --- Streamlit App UI and Logic ---
st.set_page_config(layout="wide", page_title="Holistic Health Screener")
st.title("🌿 Holistic Health Insight Assistant 🌿")

# Sidebar for API Key and Model Configuration
st.sidebar.header("AI Configuration")
st.sidebar.caption("Enter your OpenRouter API key if it's not set in secrets.")
user_api_key_input = st.sidebar.text_input(
    "OpenRouter API Key", 
    type="password", 
    key="api_key_input_sb", 
    value=OPENROUTER_API_KEY_SECRET, # Pre-fill from secrets if available
    help="Get your API key from OpenRouter.ai"
)
user_llm_model_input = st.sidebar.text_input(
    "LLM Model", 
    value=LLM_MODEL_SECRET, # Pre-fill from secrets
    key="llm_model_input_sb",
    help="e.g., mistralai/mistral-7b-instruct:free or gpt-3.5-turbo"
)

# Use sidebar inputs if provided, otherwise stick to secrets/defaults
ACTIVE_OPENROUTER_API_KEY = user_api_key_input if user_api_key_input else OPENROUTER_API_KEY_SECRET
ACTIVE_LLM_MODEL = user_llm_model_input if user_llm_model_input else LLM_MODEL_SECRET

if not ACTIVE_OPENROUTER_API_KEY:
    st.warning("OpenRouter API Key is not configured. Please add it to Streamlit secrets or enter it in the sidebar to use the AI features.")
st.sidebar.markdown(f"**Using Model:** `{ACTIVE_LLM_MODEL}`")
st.sidebar.markdown("---")
st.sidebar.header("About this Tool")
st.sidebar.info(
    """
    This tool uses a Large Language Model (AI) to help you gather information about your health.
    - The AI will ask you questions conversationally.
    - Answer honestly and thoroughly.
    - When you feel you've shared enough, type "END ASSESSMENT" in the chat.
    - A summary will then be generated for you to review and share with your doctor.
    **This tool does NOT provide medical advice or diagnosis.** All information requires professional medical interpretation.
    """
)
st.sidebar.markdown("---")
if st.sidebar.button("Clear Chat & Restart Assessment"):
    st.session_state.messages = []
    st.session_state.assessment_complete = False
    st.session_state.summary_generated = False
    st.session_state.summary_text = ""
    st.rerun()


# Initialize chat session state variables
if "messages" not in st.session_state:
    st.session_state.messages = [] # Stores {"role": "user/assistant", "content": "message text"}
if "assessment_complete" not in st.session_state:
    st.session_state.assessment_complete = False
if "summary_generated" not in st.session_state:
    st.session_state.summary_generated = False
if "summary_text" not in st.session_state:
    st.session_state.summary_text = ""

# Display initial greeting from LLM if chat is empty and API key is available
if not st.session_state.messages and ACTIVE_OPENROUTER_API_KEY:
    with st.chat_message("assistant"):
        st.markdown("Initiating conversation with the Holistic Health Insight Assistant...")
    
    # The first call to the LLM. It will use the SYSTEM_PROMPT to generate its opening.
    initial_greeting = get_llm_response(
        current_api_key=ACTIVE_OPENROUTER_API_KEY,
        current_llm_model=ACTIVE_LLM_MODEL,
        conversation_history=[{"role": "system", "content": SYSTEM_PROMPT}], 
    )
    if initial_greeting:
        st.session_state.messages.append({"role": "assistant", "content": initial_greeting})
    else:
        # Fallback greeting if API call fails for the initial message
        st.session_state.messages.append({"role": "assistant", "content": "Hello! I'm the Holistic Health Insight Assistant. To begin, could you please tell me about your main health concerns or symptoms you'd like to discuss today?"})
    st.rerun() # Rerun to display the greeting immediately

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Main chat input logic
if not st.session_state.assessment_complete and ACTIVE_OPENROUTER_API_KEY:
    if prompt := st.chat_input("What would you like to share? (Type 'END ASSESSMENT' when done)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Check if user wants to end the assessment
        # Condition for ending: "END ASSESSMENT" is in the prompt and there's some conversation history
        if "END ASSESSMENT" in prompt.upper() and len(st.session_state.messages) > 2: # Ensure there's some conversation
            st.session_state.assessment_complete = True
            st.success("Thank you. The information gathering is complete. Click the button below to generate the summary for your doctor.")
            st.rerun() # Rerun to show the "Generate Summary" button
        else:
            # Prepare the full conversation history for the LLM, including the system prompt
            current_conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.messages
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response_text = get_llm_response(
                        current_api_key=ACTIVE_OPENROUTER_API_KEY,
                        current_llm_model=ACTIVE_LLM_MODEL,
                        conversation_history=current_conversation_history
                    )
                    if response_text:
                        st.markdown(response_text)
                        st.session_state.messages.append({"role": "assistant", "content": response_text})
                    else:
                        st.markdown("Sorry, I encountered an issue. Please ensure your API key is correct and try again.")
                        # Optionally, remove the last user message if the assistant fails, or allow retry.

elif not ACTIVE_OPENROUTER_API_KEY:
    st.info("Please enter a valid OpenRouter API key in the sidebar to begin the assessment.")

# Button to generate summary
if st.session_state.assessment_complete and not st.session_state.summary_generated and ACTIVE_OPENROUTER_API_KEY:
    if st.button("Generate Doctor's Summary", type="primary"):
        with st.spinner("Generating summary for your doctor... This may take a moment."):
            # Prepare conversation text for the summarization prompt
            # Exclude the initial system prompt from the text to be summarized, but keep user/assistant turns
            conversation_for_summary = "\n".join(
                [f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages]
            )
            
            summarization_full_prompt_for_llm = SUMMARIZATION_PROMPT_TEMPLATE.format(
                conversation_history_text=conversation_for_summary
            )
            
            # For summarization, the LLM is given a new "persona" or instruction set via the prompt
            # The summarization prompt itself acts as the main instruction.
            summary_payload_messages = [
                # A brief system prompt can guide the summarizer persona, if needed by the model.
                # Or the main summarization prompt can be a single user message.
                # For many models, providing the detailed instructions as a "user" message works well.
                {"role": "system", "content": "You are an AI assistant specialized in summarizing patient conversations into structured medical reports for doctors. Follow the user's instructions precisely for the report format."},
                {"role": "user", "content": summarization_full_prompt_for_llm}
            ]

            summary_text = get_llm_response(
                current_api_key=ACTIVE_OPENROUTER_API_KEY,
                current_llm_model=ACTIVE_LLM_MODEL, # Consider if a different model is better for summarization
                conversation_history=summary_payload_messages
            )

            if summary_text:
                st.session_state.summary_text = summary_text
                st.session_state.summary_generated = True
                st.success("Summary generated successfully!")
                st.rerun() # Rerun to display the summary section
            else:
                st.error("Could not generate the summary. Please check your API key and model settings, then try again.")

# Display the generated summary
if st.session_state.summary_generated:
    st.markdown("---")
    st.subheader("📋 Summary for Your Doctor")
    st.markdown(
        "Please review this summary carefully. You can copy it or download it to share with your doctor. "
        "Remember, this is based on your self-reported information and is not a diagnosis."
    )
    st.text_area("Doctor's Summary:", value=st.session_state.get("summary_text", "Error: Summary not available."), height=500, key="summary_display_area")
    st.download_button(
        label="Download Summary as Text File",
        data=st.session_state.get("summary_text", "").encode("utf-8"),
        file_name="holistic_health_summary.txt",
        mime="text/plain"
    )
