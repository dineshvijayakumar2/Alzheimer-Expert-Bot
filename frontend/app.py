import streamlit as st
import requests
import pandas as pd
from typing import List, Dict
from datetime import datetime
import os
import time

# Custom CSS styling
st.set_page_config(page_title="Alzheimer Expert Bot", layout="wide")

st.markdown("""
<style>
    .main {
        padding: 2rem;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f8fafc;
        border-radius: 4px;
        padding: 10px 20px;
        font-weight: 500;
        color: #1a202c;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #e2e8f0;
    }

    /* Container styling */
    .chat-container {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-top: 1rem;
    }
    .response-container {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #e2e8f0;
    }
    .source-container {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        color: #e2e8f0;
    }

    /* Button styling */
    .suggestion-button {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 0.5rem 1rem;
        border-radius: 4px;
        margin: 0.25rem;
        cursor: pointer;
        color: #e2e8f0;
    }

    /* Resource card styling */
    .resource-card {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 0.5rem 0;
        color: #e2e8f0;
    }

    /* Text styling for dark mode */
    .stMarkdown {
        color: #e2e8f0 !important;
    }
    .stMarkdown p {
        color: #e2e8f0 !important;
    }
    .stMarkdown strong {
        color: #f8fafc !important;
    }
    .stMarkdown a {
        color: #60a5fa !important;
    }

    /* Chat history styling */
    .chat-message {
        color: #e2e8f0 !important;
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    .chat-message-user {
        background-color: rgba(59, 130, 246, 0.1);
    }
    .chat-message-assistant {
        background-color: rgba(255, 255, 255, 0.05);
    }

    /* Expander styling */
    .stExpander {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
    }
    .stExpander p {
        color: #e2e8f0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'query' not in st.session_state:
    st.session_state.query = ''
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'search_mode' not in st.session_state:
    st.session_state.search_mode = 'resource'
if 'api_url' not in st.session_state:
    st.session_state.api_url = os.getenv('API_URL', 'http://backend:8000')
if 'user_input' not in st.session_state:
    st.session_state.user_input = ''
if 'rerun_counter' not in st.session_state:
    st.session_state.rerun_counter = 0

def create_message_payload(message: str, history: List[Dict[str, str]], search_mode: str) -> Dict:
    """Create the payload for the chat API request"""
    return {
        "message": message,
        "history": history,
        "search_mode": search_mode
    }


def handle_suggested_question(question: str):
    """Handle when a suggested question is clicked"""
    st.session_state.query = question  # Set the query for next render
    st.session_state.last_input = question  # Prevent duplicate messages
    st.session_state.rerun_counter += 1
    handle_user_input(question)


def handle_chat_response(response_data: Dict) -> None:
    """Handle and display the chat response with formatting"""
    with st.markdown("""
        <div class="response-container">
            <h4 style="color: #f8fafc;">Clinical Response:</h4>
        </div>
    """, unsafe_allow_html=True):
        st.markdown(f'<div class="chat-message chat-message-assistant">{response_data["response"]}</div>',
                   unsafe_allow_html=True)

    if response_data.get("sources"):
        with st.expander("📚 Reference Sources", expanded=False):
            for source in response_data["sources"]:
                st.markdown(f"""
                    <div class="source-container">
                        <p style="color: #f8fafc;"><strong>{source['title']}</strong></p>
                        <p style="color: #e2e8f0;">{source['ieee_citation']}</p>
                    </div>
                """, unsafe_allow_html=True)

    if response_data.get("suggested_questions"):
        st.markdown("### Related Clinical Queries:")
        cols = st.columns(len(response_data["suggested_questions"]))
        for i, (question, col) in enumerate(zip(response_data["suggested_questions"], cols)):
            with col:
                if st.button(
                    question,
                    key=f"suggest_{i}_{st.session_state.rerun_counter}",
                    help="Click to ask this question",
                    use_container_width=True
                ):
                    st.session_state.query = question
                    handle_suggested_question(question)

def handle_user_input(user_input: str) -> None:
    """Process user input and update chat history"""
    if user_input and (not st.session_state.get('last_input') == user_input):
        st.session_state.last_input = user_input

        try:
            # Add user message to history first
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })

            # Make API request
            response = requests.post(
                f"{st.session_state.api_url}/chat",
                json=create_message_payload(
                    message=user_input,
                    history=st.session_state.chat_history,
                    search_mode=st.session_state.search_mode
                )
            )
            response.raise_for_status()
            response_data = response.json()

            # Add assistant response to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response_data["response"]
            })

            # Handle and display the response
            handle_chat_response(response_data)

        except requests.exceptions.RequestException as e:
            st.error(f"Error communicating with the server: {str(e)}")


def display_chatbot():
    """Display the chatbot interface"""
    st.header("🤖 Clinical Decision Support")

    # Store previous input before radio change
    previous_input = st.session_state.get(f'query_input_{st.session_state.rerun_counter}', '')

    # Search mode selection
    search_mode = st.radio(
        "Select Search Mode:",
        ["Resource-Based Search", "Open Clinical Search"],
        key="search_mode_radio",
        help="Choose between searching through curated medical resources or open-ended clinical queries"
    )

    # Update search mode without clearing input
    new_mode = 'resource' if search_mode == "Resource-Based Search" else 'open'
    if st.session_state.search_mode != new_mode:
        st.session_state.search_mode = new_mode
        # Preserve the previous input
        st.session_state[f'query_input_{st.session_state.rerun_counter}'] = previous_input

    # Mode information
    if st.session_state.search_mode == 'resource':
        st.info("📚 Resource-Based Search: Using curated medical literature with citations")
    else:
        st.info("🔍 Open Search: Accessing broader medical knowledge base")

    # Chat interface
    with st.container():
        user_input = st.text_area(
            "Enter your clinical query:",
            value=previous_input,  # Use the preserved input
            height=100,
            key=f'query_input_{st.session_state.rerun_counter}'
        )

        if st.button("Submit Query", type="primary", key=f"submit_{st.session_state.rerun_counter}"):
            if user_input:
                handle_user_input(user_input)
                st.session_state.rerun_counter += 1

        if st.session_state.chat_history:
            st.markdown("### Consultation History")
            for message in st.session_state.chat_history:
                role = "🩺 Assistant:" if message["role"] == "assistant" else "👤 You:"
                message_class = "chat-message-assistant" if message["role"] == "assistant" else "chat-message-user"
                st.markdown(f"""
                    <div class="chat-message {message_class}">
                        <strong style="color: #f8fafc;">{role}</strong>
                        <div style="margin-top: 0.5rem;">{message["content"]}</div>
                    </div>
                """, unsafe_allow_html=True)


def display_resource_upload():
    """Display resource upload interface"""
    st.header("📤 Add New Resource")

    # Resource type selection
    resource_type = st.selectbox(
        "Resource Type",
        ["url", "pdf", "text"],
        help="Select the type of resource you want to add"
    )

    # Resource title
    title = st.text_input("Title", help="Enter a descriptive title for the resource")

    # Tags
    tags = st.text_input(
        "Tags (comma-separated)",
        help="Enter tags to categorize the resource"
    ).split(",")
    tags = [tag.strip() for tag in tags if tag.strip()]

    # Resource content based on type
    url = None
    content = None
    file_path = None

    if resource_type == "url":
        url = st.text_input("URL", help="Enter the webpage URL")
    elif resource_type == "pdf":
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        if uploaded_file:
            os.makedirs("temp", exist_ok=True)
            file_path = f"temp/{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
    else:  # text
        content = st.text_area("Content", help="Enter or paste the text content")

    if st.button("Add Resource"):
        if not title:
            st.error("Title is required")
            return

        # Validate input based on resource type
        if resource_type == "url" and not url:
            st.error("URL is required for URL resource type")
            return
        elif resource_type == "text" and not content:
            st.error("Content is required for text resource type")
            return
        elif resource_type == "pdf" and not file_path:
            st.error("PDF file is required for PDF resource type")
            return

        try:
            with st.spinner("Adding resource..."):
                # Prepare the resource data according to ResourceCreate model
                resource_data = {
                    "title": title,
                    "type": resource_type,
                    "tags": tags,
                }

                if resource_type == "url":
                    resource_data["url"] = url
                    resource_data["content"] = f"URL Resource: {url}"
                elif resource_type == "text":
                    resource_data["content"] = content

                # For PDF handling
                if resource_type == "pdf":
                    try:
                        with open(file_path, "rb") as pdf_file:
                            files = {"file": ("file.pdf", pdf_file, "application/pdf")}
                            response = requests.post(
                                f"{st.session_state.api_url}/resources/upload-pdf",
                                data=resource_data,
                                files=files
                            )
                    except FileNotFoundError:
                        st.error("Error accessing the uploaded PDF file")
                        return
                else:
                    # For URL and text resources
                    response = requests.post(
                        f"{st.session_state.api_url}/resources/add",
                        json=resource_data
                    )

                # Handle response
                if response.ok:
                    result = response.json()
                    if result.get("status") == "success":
                        st.success("Resource added successfully!")
                        st.rerun()
                    else:
                        st.error(f"Error: {result.get('message', 'Unknown error')}")
                else:
                    error_detail = response.json().get('detail', 'Unknown error')
                    st.error(f"Error: {error_detail}")

        except requests.exceptions.RequestException as e:
            st.error(f"Network error: {str(e)}")
        except Exception as e:
            st.error(f"Error uploading resource: {str(e)}")
        finally:
            # Clean up temporary files
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    st.error(f"Error cleaning up temporary file: {str(e)}")

def handle_pdf_upload(file_path: str, resource_data: dict) -> requests.Response:
    """Handle PDF file upload with proper error handling"""
    try:
        with open(file_path, "rb") as pdf_file:
            files = {"file": pdf_file}
            response = requests.post(
                f"{st.session_state.api_url}/resources/upload-pdf",
                data=resource_data,
                files=files
            )
        return response
    except FileNotFoundError:
        raise Exception("PDF file not found or inaccessible")
    except Exception as e:
        raise Exception(f"Error uploading PDF: {str(e)}")


def display_resource_list():
    """Display list of resources"""
    st.header("📚 Existing Resources")

    try:
        response = requests.get(f"{st.session_state.api_url}/resources")
        if response.ok:
            resources = response.json()
            if not resources:
                st.info("No resources found")
                return

            # Convert to DataFrame and handle the data structure properly
            df = pd.DataFrame(resources)

            # Check if the data is nested in metadata
            if 'metadata' in df.columns:
                # Extract metadata fields
                metadata_df = pd.json_normalize(df['metadata'])
                # Combine with main dataframe
                df = pd.concat([df.drop('metadata', axis=1), metadata_df], axis=1)

            # Clean up and rename columns
            column_mapping = {
                "_id": "ID",
                "title": "Title",
                "type": "Type",
                "date_added": "Added",
                "added_at": "Added",  # Alternative field name
                "tags": "Tags"
            }

            # Only rename columns that exist
            df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

            # Select and reorder columns that exist
            available_columns = []
            for col in ["ID", "Title", "Type", "Added", "Tags"]:
                if col in df.columns:
                    available_columns.append(col)

            df = df[available_columns]

            # Format date if it exists
            if "Added" in df.columns:
                try:
                    df["Added"] = pd.to_datetime(df["Added"]).dt.strftime("%Y-%m-%d %H:%M")
                except:
                    pass

            # Format tags if they exist
            if "Tags" in df.columns:
                df["Tags"] = df["Tags"].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)

            # Display as table with custom styling
            st.dataframe(
                df,
                use_container_width=True,
                column_config={
                    "ID": st.column_config.Column(width="medium"),
                    "Title": st.column_config.Column(width="large"),
                    "Type": st.column_config.Column(width="small"),
                    "Added": st.column_config.Column(width="medium"),
                    "Tags": st.column_config.Column(width="medium")
                }
            )

            # Delete resource section
            if len(df) > 0:
                with st.expander("Delete Resource"):
                    if "ID" in df.columns and "Title" in df.columns:
                        resource_id = st.selectbox(
                            "Select Resource to Delete",
                            options=df["ID"].tolist(),
                            format_func=lambda x: df[df["ID"] == x]["Title"].iloc[0]
                        )
                        if st.button("Delete Selected Resource", type="primary"):
                            if resource_id:
                                with st.spinner("Deleting resource..."):
                                    response = requests.delete(
                                        f"{st.session_state.api_url}/resources/{resource_id}"
                                    )
                                    if response.ok:
                                        st.success("Resource deleted successfully!")
                                        st.rerun()
                                    else:
                                        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
        else:
            st.error(f"Error fetching resources: {response.json().get('detail', 'Unknown error')}")

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.error("Raw response data for debugging:")
        try:
            st.json(resources)
        except:
            st.write("Could not display raw data")

def display_resource_search():
    """Display resource search interface"""
    st.header("🔍 Search Resources")

    # Search inputs
    query = st.text_input("Search Query")
    resource_types = st.multiselect(
        "Resource Types",
        ["Clinical", "Research", "Guidelines"],
        default=["Clinical", "Research", "Guidelines"]
    )

    if query or st.button("Show All Resources"):
        with st.spinner("Searching resources..."):
            try:
                params = {
                    "query": query if query else None,
                    "types": resource_types
                }
                response = requests.get(
                    f"{st.session_state.api_url}/resources/search",
                    params=params
                )
                response.raise_for_status()
                results = response.json()

                if results:
                    for idx, result in enumerate(results, 1):
                        with st.expander(f"Result {idx}: {result.get('title', 'Untitled')}"):
                            # Content section
                            st.markdown("**Content:**")
                            content = result.get('content', '')
                            st.markdown(content[:300] + "..." if len(content) > 300 else content)

                            # Metadata section
                            st.markdown("---")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"Type: {result.get('type', 'Unknown')}")
                                tags = result.get('tags', [])
                                st.write(f"Tags: {', '.join(tags) if tags else 'No tags'}")
                            with col2:
                                st.write(f"Added: {result.get('date_added', 'Unknown date')}")
                                if url := result.get('url'):
                                    st.write(f"URL: [{url}]({url})")
                else:
                    st.info("No results found")

            except Exception as e:
                st.error(f"Error searching resources: {str(e)}")


def display_connection_test():
    """Display connection test interface"""
    st.header("🔧 Connection Test")

    if st.button("Test Connection"):
        with st.spinner("Testing connection to backend services..."):
            try:
                response = requests.get(f"{st.session_state.api_url}/health")
                if response.ok:
                    data = response.json()
                    st.success("✅ Backend services are healthy")

                    # Display component status
                    components = data.get('components', {})
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("MongoDB:", "🟢 Connected" if components.get('mongodb') else "🔴 Disconnected")
                        st.write("Vector Store:", "🟢 Active" if components.get('vector_store') else "🔴 Inactive")
                    with col2:
                        st.write("Embeddings:", "🟢 Working" if components.get('embeddings') else "🔴 Not Working")
                        st.write("Anthropic:", "🟢 Connected" if components.get('anthropic') else "🔴 Disconnected")
                else:
                    st.error("❌ Backend services are unhealthy")
            except Exception as e:
                st.error(f"❌ Connection test failed: {str(e)}")


def display_resource_management():
    """Display resource management interface"""
    st.header("📚 Resource Management")

    # Tabs for different resource management functions
    tabs = st.tabs(["Connection Test", "Add Resource", "View Resources", "Search Resources"])

    with tabs[0]:
        display_connection_test()
    with tabs[1]:
        display_resource_upload()
    with tabs[2]:
        display_resource_list()
    with tabs[3]:
        display_resource_search()


def main():
    st.title("🧠 Alzheimer Expert Bot")

    # Main navigation
    tabs = st.tabs(["Clinical Assistant", "Resource Management"])

    with tabs[0]:
        display_chatbot()
    with tabs[1]:
        display_resource_management()

    # Footer
    st.markdown("---")
    st.markdown("*Powered by Advanced AI for Clinical Decision Support*")


if __name__ == "__main__":
    main()