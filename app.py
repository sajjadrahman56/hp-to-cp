import gradio as gr
import PyPDF2
import io
import os
from together import Together

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file"""
    text = ""
    try:
        # Check if the pdf_file is already in bytes format or needs conversion
        if hasattr(pdf_file, 'read'):
            # If it's a file-like object (from gradio upload)
            pdf_content = pdf_file.read()
            # Reset the file pointer for potential future reads
            if hasattr(pdf_file, 'seek'):
                pdf_file.seek(0)
        else:
            # If it's already bytes
            pdf_content = pdf_file
            
        # Read the PDF file
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        
        # Extract text from each page
        for page_num in range(len(pdf_reader.pages)):
            page_text = pdf_reader.pages[page_num].extract_text()
            if page_text:  # Check if text extraction worked
                text += page_text + "\n\n"
            else:
                text += f"[Page {page_num+1} - No extractable text found]\n\n"
        
        if not text.strip():
            return "No text could be extracted from the PDF. The document may be scanned or image-based."
            
        return text
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

def format_chat_history(history):
    """Format the chat history for display"""
    formatted_history = []
    for user_msg, bot_msg in history:
        formatted_history.append((user_msg, bot_msg))
    return formatted_history

def chat_with_pdf(api_key, pdf_text, user_question, history):
    """Chat with the PDF using Together API"""
    if not api_key.strip():
        return history + [(user_question, "Error: Please enter your Together API key.")], history
    
    if not pdf_text.strip() or pdf_text.startswith("Error") or pdf_text.startswith("No text"):
        return history + [(user_question, "Error: Please upload a valid PDF file with extractable text first.")], history
    
    if not user_question.strip():
        return history + [(user_question, "Error: Please enter a question.")], history
    
    try:
        # Initialize Together client with the API key
        client = Together(api_key=api_key)
        
        # Create the system message with PDF context
        # Truncate the PDF text if it's too long (model context limit handling)
        max_context_length = 10000 #10000
        
        if len(pdf_text) > max_context_length:
            # More sophisticated truncation that preserves beginning and end
            half_length = max_context_length // 2
            pdf_context = pdf_text[:half_length] + "\n\n[...Content truncated due to length...]\n\n" + pdf_text[-half_length:]
        else:
            pdf_context = pdf_text
        
        system_message = f"""You are an intelligent assistant designed to read, understand, and extract information from PDF documents. 
Based on any question or query the user asks—whether it's about content, summaries, data extraction, definitions, insights, or interpretation—you will 
analyze the following PDF content and provide an accurate, helpful response grounded in the document. Always respond with clear, concise, and context-aware information.
PDF CONTENT:
{pdf_context}
Answer the user's questions only based on the PDF content above. If the answer cannot be found in the PDF, politely state that the information is not available in the provided document."""
        
        # Prepare message history for Together API
        messages = [
            {"role": "system", "content": system_message},
        ]
        
        # Add chat history
        for h_user, h_bot in history:
            messages.append({"role": "user", "content": h_user})
            messages.append({"role": "assistant", "content": h_bot})
        
        # Add the current user question
        messages.append({"role": "user", "content": user_question})
        
        # Call the Together API
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=messages,
            max_tokens=5000, #5000
            temperature=0.7,
        )
        
        # Extract the assistant's response
        assistant_response = response.choices[0].message.content
        
        # Update the chat history
        new_history = history + [(user_question, assistant_response)]
        
        return new_history, new_history
    
    except Exception as e:
        error_message = f"Error: {str(e)}"
        return history + [(user_question, error_message)], history

def process_pdf(pdf_file, api_key_input):
    """Process the uploaded PDF file"""
    if pdf_file is None:
        return "Please upload a PDF file.", "", []
    
    try:
        # Get the file name
        file_name = os.path.basename(pdf_file.name) if hasattr(pdf_file, 'name') else "Uploaded PDF"
        
        # Extract text from the PDF
        pdf_text = extract_text_from_pdf(pdf_file)
        
        # Check if there was an error in extraction
        if pdf_text.startswith("Error extracting text from PDF"):
            return f"❌ {pdf_text}", "", []
        
        if not pdf_text.strip() or pdf_text.startswith("No text could be extracted"):
            return f"⚠️ {pdf_text}", "", []
        
        # Count words for information
        word_count = len(pdf_text.split())
        
        # Return a message with the file name and text content
        status_message = f"✅ Successfully processed PDF: {file_name} ({word_count} words extracted)"
        
        # Also return an empty history
        return status_message, pdf_text, []
    except Exception as e:
        return f"❌ Error processing PDF: {str(e)}", "", []

def validate_api_key(api_key):
    """Simple validation for API key format"""
    if not api_key or not api_key.strip():
        return "❌ API Key is required"
    
    if len(api_key.strip()) < 10:
        return "❌ API Key appears to be too short"
    
    return "✓ API Key format looks valid (not verified with server)"

# Create the Gradio interface
with gr.Blocks(title="ChatPDF with Together AI") as app:
    gr.Markdown("# 📄 ChatPDF with Together AI")
    gr.Markdown("Upload a PDF and chat with it using the Llama-3.3-70B model.")
    
    with gr.Row():
        with gr.Column(scale=1):
            # API Key input
            api_key_input = gr.Textbox(
                label="Together API Key",
                placeholder="Enter your Together API key here...",
                type="password"
            )
            
            # API key validation
            api_key_status = gr.Textbox(
                label="API Key Status",
                interactive=False
            )
            
            # PDF upload
            pdf_file = gr.File(
                label="Upload PDF",
                file_types=[".pdf"],
                type="binary"  # Ensure we get binary data
            )
            
            # Process PDF button
            process_button = gr.Button("Process PDF")
            
            # Status message
            status_message = gr.Textbox(
                label="Status",
                interactive=False
            )
            
            # Hidden field to store the PDF text
            pdf_text = gr.Textbox(visible=False)
            
            # Optional: Show PDF preview
            with gr.Accordion("PDF Content Preview", open=False):
                pdf_preview = gr.Textbox(
                    label="Extracted Text Preview",
                    interactive=False,
                    max_lines=10,
                    show_copy_button=True
                )
        
        with gr.Column(scale=2):
            # Chat interface
            chatbot = gr.Chatbot(
                label="Chat with PDF",
                height=500
            )
            
            # Question input
            question = gr.Textbox(
                label="Ask a question about the PDF",
                placeholder="What is the main topic of this document?",
                lines=2
            )
            
            # Submit button
            submit_button = gr.Button("Submit Question")
    
    # Event handlers
    def update_preview(text):
        """Update the preview with the first few lines of the PDF text"""
        if not text or text.startswith("Error") or text.startswith("No text"):
            return text
        
        # Get the first ~500 characters for preview
        preview = text[:500]
        if len(text) > 500:
            preview += "...\n[Text truncated for preview. Full text will be used for chat.]"
        return preview
    
    # API key validation event
    api_key_input.change(
        fn=validate_api_key,
        inputs=[api_key_input],
        outputs=[api_key_status]
    )
    
    process_button.click(
        fn=process_pdf,
        inputs=[pdf_file, api_key_input],
        outputs=[status_message, pdf_text, chatbot]
    ).then(
        fn=update_preview,
        inputs=[pdf_text],
        outputs=[pdf_preview]
    )
    
    submit_button.click(
        fn=chat_with_pdf,
        inputs=[api_key_input, pdf_text, question, chatbot],
        outputs=[chatbot, chatbot]
    ).then(
        fn=lambda: "",
        outputs=question
    )
    
    question.submit(
        fn=chat_with_pdf,
        inputs=[api_key_input, pdf_text, question, chatbot],
        outputs=[chatbot, chatbot]
    ).then(
        fn=lambda: "",
        outputs=question
    )

# Launch the app
if __name__ == "__main__":
    app.launch(share=True)
