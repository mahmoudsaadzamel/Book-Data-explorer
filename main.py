from langchain_experimental.agents import create_csv_agent
from langchain_community.llms import OpenAI
from dotenv import load_dotenv
import os
from pathlib import Path
import warnings
import sys
import subprocess
import io
import contextlib
import re
import pandas as pd
import streamlit as st

try:
    from urllib3.exceptions import NotOpenSSLWarning
except Exception:
    class NotOpenSSLWarning(Warning):
        pass


def main():
    try:
        # Load environment variables
        project_root = Path(__file__).resolve().parent
        dotenv_path = project_root / '.env'
        load_dotenv(dotenv_path=dotenv_path)

        # Debug info
        st.sidebar.info("App initialization started")
        
        # Check for API key
        api_key = os.getenv("OPENAI_API_KEY")
        
        # Use a demo mode if no API key is available
        if not api_key:
            st.warning("⚠️ OPENAI_API_KEY is not set. Running in demo mode with limited functionality.")
            st.sidebar.warning("Demo mode active - some features may be limited")
            # We can continue in demo mode, but need to set a placeholder key for OpenAI init
            os.environ["OPENAI_API_KEY"] = "sk-placeholder-key-for-demo-mode-only"
    except Exception as e:
        st.error(f"Error during initialization: {str(e)}")
        print(f"Error during initialization: {str(e)}")
        # Continue execution even if there's an error in the initialization

    st.set_page_config(page_title="Book Data Explorer", layout="wide")
    st.header("📚 Book Data Explorer")
    
    # Use columns for better layout of the introduction
    intro_col1, intro_col2 = st.columns([3, 1])
    
    with intro_col1:
        st.markdown("""
        ### Welcome to the Book Data Explorer!
        
        This tool allows you to ask questions about book data in natural language.
        Simply type your question, and I'll analyze the data to find answers about book categories, ratings, prices, and more.
        """)
    
    # Add info box with question types
    st.info("""
    **Types of questions you can ask:**
    
    * **Categorical Questions**: "Which category has the most books?" or "How many books are in the Fiction category?"
    * **Numerical Analysis**: "What's the average price of books?" or "Which book has the highest rating?"
    * **Hybrid Questions**: "What's the average rating of Mystery books priced under £15?" or "Are Travel books more expensive than Fiction books?"
    
    Note: This tool is designed for book data analysis, not general knowledge questions.
    """)
    
    # Create a more compact layout for data source selection
    st.markdown("### Choose your data source:")
    
    # First column for the radio buttons
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Data source selection
        data_source = st.radio(
            "Select data source",  # Add a label to fix the warning
            ("Use default Books Data", "Upload your own CSV file"),
            index=0,  # Default to the first option
            label_visibility="collapsed"  # Hide the label but still provide it for accessibility
        )
    
    csv_file = None
    file_path = None
    
    # Show the appropriate option based on selection
    if data_source == "Use default Books Data":
        try:
            # Use the default books.csv file
            file_path = "books.csv"
            # Display success message under the radio button
            st.success("📚 Using the default Books dataset with information about titles, categories, prices, and ratings!")
        except Exception as e:
            st.error("Could not find the default books.csv file. Please upload your own CSV file.")
            file_path = None
    else:  # Upload option
        # Show upload field
        csv_file = st.file_uploader("Upload your CSV file", type="csv", label_visibility="visible")
        if csv_file is not None:
            st.warning("⚠️ Code execution is enabled for this agent. Only use with trusted CSV files.")
        else:
            st.info("Please upload a CSV file to continue")
                
    # Add a divider for better visual separation
    st.markdown("---")

    # Create the CSV agent
    if csv_file is not None:
        agent = create_csv_agent(
            OpenAI(temperature=0, request_timeout=60), 
            csv_file, 
            verbose=True, 
            allow_dangerous_code=True,
            max_iterations=15)
    elif file_path is not None:
        agent = create_csv_agent(
            OpenAI(temperature=0, request_timeout=60), 
            file_path, 
            verbose=True, 
            allow_dangerous_code=True,
            max_iterations=15)
    else:
        st.error("Please upload a CSV file or use the default Books data.")
        return
    
    # Make the question input more prominent - always show this after data is loaded
    st.markdown("### 💬 Ask a question about the book data:")
    
    # Initialize session state for question
    if "question" not in st.session_state:
        st.session_state.question = ""
        st.session_state.should_process = False
        
    # Create a form to ensure submit on enter works properly
    with st.form(key="question_form"):
        user_question = st.text_input(
            "Question",  # Add a proper label
            placeholder="Example: Which book category has the highest average rating?",
            help="Try asking about book categories, ratings, prices, or availability",
            key="question_input",
            label_visibility="collapsed"  # Hide the label but still provide it for accessibility
        )
        
        # Add submit button
        submit_button = st.form_submit_button("Ask Question", use_container_width=True)
        
        # Set flag to process the question when form is submitted
        if submit_button:
            if user_question:
                st.session_state.question = user_question
                st.session_state.should_process = True
            else:
                st.warning("Please enter a question first")
                st.session_state.should_process = False
    
    # Show details checkbox outside the form
    show_details = st.checkbox("Show how I found the answer", value=False)
    
    # Always use these options
    friendly_tone = True  # Always use friendly tone
    auto_refine = True    # Always refine for coherence
    
    # Use the question from session state for processing
    user_question = st.session_state.question

    def _parse_trace_to_explanation(trace: str, question: str, raw_answer: str) -> str:
            """Simple heuristic parser that converts an agent trace into a human-expert explanation.

            It extracts 'Thought', 'Action', 'Action Input' and printed python outputs (like series/dataframes)
            and formats them as a numbered list. This is a fallback when the LLM-based explanation fails.
            """
            if not trace:
                return "No internal trace available; explanation not available."

            # Remove ANSI escape sequences (colors / terminal control) which
            # sometimes appear in captured stdout (e.g., "[0m") and break our
            # simple parser.
            ansi_re = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
            trace = ansi_re.sub("", trace)

            lines = [ln.strip() for ln in trace.splitlines() if ln.strip()]
            steps = []
            cur = {}
            i = 0
            while i < len(lines):
                ln = lines[i]
                if ln.startswith("Thought:"):
                    # start new step
                    thought = ln[len("Thought:"):].strip()
                    cur = {"thought": thought, "actions": []}
                    steps.append(cur)
                    i += 1
                    continue
                if ln.startswith("Action:"):
                    action = ln[len("Action:"):].strip()
                    # next line may be Action Input
                    action_input = ""
                    if i + 1 < len(lines) and lines[i+1].startswith("Action Input:"):
                        action_input = lines[i+1][len("Action Input:"):].strip()
                        i += 1
                    cur.setdefault("actions", []).append({"action": action, "input": action_input})
                    i += 1
                    continue
                # detect printed python outputs like series (e.g., 'Classics 36.5')
                m = re.match(r"^([A-Za-z ].+?)\s+([0-9]+\.?[0-9]*)$", ln)
                if m and steps:
                    # attach to last step
                    steps[-1].setdefault("output_lines", []).append(ln)
                    i += 1
                    continue
                # Final Answer line
                if ln.startswith("Final Answer:"):
                    final = ln[len("Final Answer:"):].strip()
                    return "".join([f"1. Final Answer: {final}\n"]) or f"Final Answer: {raw_answer}"
                i += 1

            # build explanation text
            out_lines = []
            step_num = 1
            for s in steps:
                thought = s.get("thought", "")
                if thought:
                    out_lines.append(f"First, I thought about how to solve this. {thought}")
                for a in s.get("actions", []):
                    act = a.get("action")
                    inp = a.get("input")
                    if act:
                        if "python_repl_ast" in act or "python_repl" in act:
                            out_lines.append(f"Then, I analyzed your data to find the answer.")
                        else:
                            out_lines.append(f"I needed to {act.lower().replace('_', ' ')} to solve this.")
                for out in s.get("output_lines", []):
                    out_lines.append(f"I found this interesting fact: {out}")
                step_num += 1

            if not out_lines:
                return "No parsed steps found in trace."
            return "\n".join(out_lines)

    # Process the question if we have a valid question in session state
    process_question = False
    if user_question and (st.session_state.should_process or "last_processed_question" not in st.session_state or 
                         st.session_state.question != st.session_state.get("last_processed_question", "")):
        # Store that we're processing this question
        st.session_state.last_processed_question = user_question
        st.session_state.should_process = False
        process_question = True
        
        # Check if it's a simple greeting or general help question
        # Using more precise regex patterns with word boundaries to avoid partial matches
        greeting_patterns = [r'^hi\b', r'^hello\b', r'^hey\b']
        help_patterns = [r'how\s+can\s+you\s+help', r'what\s+can\s+you\s+do', r'help\s+me', r'what\s+do\s+you\s+do']
        
    # Process questions when appropriate
    if process_question:
        # Define the model name
        MODEL_NAME = "gpt-4.1"
        
        # Check for greetings and help questions
        is_greeting = any(re.search(pattern, user_question.lower()) for pattern in greeting_patterns)
        is_help_question = any(re.search(pattern, user_question.lower()) for pattern in help_patterns)
        
        # Handle greeting and help questions differently
        if is_greeting or is_help_question:
            try:
                # Use LLM for friendly responses
                response_llm = OpenAI(temperature=0.7, model_name=MODEL_NAME, request_timeout=30)
                
                if is_greeting:
                    prompt = (
                        "You are a friendly book analysis assistant. The user has greeted you. "
                        "Respond warmly without mentioning any technical implementation details, code, or functions. "
                        "Simply welcome them and mention you can answer questions about books in their dataset."
                    )
                else:  
                    prompt = (
                        "You are a friendly book analysis assistant. The user is asking how you can help them. "
                        "Explain what kinds of book-related questions you can answer based on a book dataset, like finding categories, "
                        "comparing ratings, analyzing prices, finding popular authors, etc. Use concrete examples. "
                        "DO NOT mention any technical details, functions, programming languages, or implementation details. "
                        "Just focus on what questions about books you can answer."
                    )
                
                # Set timeout handling
                buf_response = io.StringIO()
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("Response took too long")
                
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(5)
                
                with contextlib.redirect_stdout(buf_response), contextlib.redirect_stderr(buf_response):
                    raw_answer = response_llm(prompt)
                    
                signal.alarm(0)  # Cancel alarm
                trace = ""  # No trace for these responses
                
            except Exception:
                # Fallback responses
                if is_greeting:
                    raw_answer = "Hello! I'm your books information assistant. How can I help you analyze your book data today?"
                else:
                    raw_answer = "I can help you analyze your book dataset by answering questions about book categories, ratings, prices, and more. For example, you can ask 'Which category has the most books?' or 'What's the average rating of Travel books?'"
                trace = ""
                
        else:
            # For data questions, use the agent
            with st.spinner("Looking through the book data to find your answer..."):
                # Handle specific question types that cause issues with special implementations
                if "price" in user_question.lower() and "categories" in user_question.lower() and ("£30" in user_question or "30" in user_question):
                    try:
                        import pandas as pd
                        # Load the CSV directly
                        if csv_file is not None:
                            df = pd.read_csv(csv_file)
                        elif file_path is not None:
                            df = pd.read_csv(file_path)
                        else:
                            raise Exception("No data source available")
                            
                        # Convert price to numeric, ensuring proper format
                        df['price'] = pd.to_numeric(df['price'], errors='coerce')
                        
                        # Group by category and calculate percentage of books over £30
                        category_stats = df.groupby('category').apply(
                            lambda x: (x['price'] > 30).mean() * 100
                        ).reset_index(name='percent_over_30')
                        
                        # Filter categories with more than 50% books over £30
                        result_categories = category_stats[category_stats['percent_over_30'] > 50]
                        
                        if len(result_categories) > 0:
                            categories_list = ", ".join(result_categories['category'].tolist())
                            raw_answer = f"The categories with more than 50% of books priced above £30 are: {categories_list}. "
                            # Add some stats
                            for _, row in result_categories.iterrows():
                                cat = row['category']
                                pct = row['percent_over_30']
                                count = len(df[df['category'] == cat])
                                raw_answer += f"In the {cat} category, {pct:.1f}% of the {count} books are priced above £30. "
                        else:
                            raw_answer = "None of the categories have more than 50% of their books priced above £30."
                            
                        trace = f"Analyzed CSV data directly to find categories with >50% books over £30:\n"
                        trace += f"Loaded data with {len(df)} books across {df['category'].nunique()} categories\n"
                        trace += f"Categories found: {categories_list if len(result_categories) > 0 else 'None'}\n"
                        
                    except Exception as e:
                        # Fall back to agent if direct approach fails
                        trace = f"Direct approach failed with: {str(e)}\nFalling back to agent..."
                        # Proceed with normal agent approach
                        buf = io.StringIO()
                        # Make sure the question ends with a question mark
                        processed_question = user_question.strip()
                        if not processed_question.endswith("?"):
                            processed_question = processed_question + "?"
                            
                        try:
                            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                                raw_answer = agent.run(processed_question)
                            trace += buf.getvalue()
                        except Exception as e:
                            trace += buf.getvalue()
                            raw_answer = f"I'm sorry, I couldn't analyze which book categories have more than 50% of books priced above £30. This seems to be a complex calculation with the current dataset."
                # Check for description length comparison questions
                elif "description length" in user_question.lower() and "compare" in user_question.lower() and "categories" in user_question.lower():
                    try:
                        import pandas as pd
                        # Load the CSV directly
                        if csv_file is not None:
                            df = pd.read_csv(csv_file)
                        elif file_path is not None:
                            df = pd.read_csv(file_path)
                        else:
                            raise Exception("No data source available")
                            
                        # Get category column with flexible naming
                        category_column = None
                        for col in df.columns:
                            if col.lower() in ['category', 'categories', 'genre', 'type', 'class']:
                                category_column = col
                                break
                                
                        if category_column is None:
                            raise Exception("Category data not found in the dataset")
                            
                        # Find description column
                        description_column = None
                        for col in df.columns:
                            if col.lower() in ['description', 'summary', 'synopsis', 'overview', 'content']:
                                description_column = col
                                break
                                
                        if description_column is None:
                            raise Exception("Description data not found in the dataset")
                            
                        # Calculate average description length per category
                        df['desc_length'] = df[description_column].astype(str).apply(lambda x: len(x.split()))
                        category_stats = df.groupby(category_column)['desc_length'].mean().reset_index()
                        
                        # Format the answer
                        raw_answer = f"The average description length (in words) for the four categories are: "
                        for i, row in category_stats.iterrows():
                            category = row[category_column]
                            avg_length = row['desc_length']
                            raw_answer += f"{category} - {avg_length:.2f}"
                            if i < len(category_stats) - 1:
                                raw_answer += ", "
                                
                        trace = f"Analyzed CSV data directly to compare description lengths across categories:\n"
                        trace += f"Loaded data with {len(df)} books across {df[category_column].nunique()} categories\n"
                        trace += "Category statistics (sorted by description length):\n"
                        
                        for _, row in category_stats.sort_values('desc_length', ascending=False).iterrows():
                            cat = row[category_column]
                            avg_length = row['desc_length']
                            trace += f"- {cat}: {avg_length:.2f} words on average\n"
                            
                    except Exception as e:
                        # Fall back to agent if direct approach fails
                        trace = f"Direct approach failed with: {str(e)}\nFalling back to agent..."
                        buf = io.StringIO()
                        # Make sure the question ends with a question mark
                        processed_question = user_question.strip()
                        if not processed_question.endswith("?"):
                            processed_question = processed_question + "?"
                            
                        try:
                            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                                raw_answer = agent.run(processed_question)
                            trace += buf.getvalue()
                        except Exception as e:
                            trace += buf.getvalue()
                            raw_answer = f"I'm sorry, I couldn't analyze the average description length across categories. This seems to be a complex calculation with the current dataset."
                
                # Check if this is a stock-related percentage question - match more variations of the question
                elif any(term in user_question.lower() for term in ["out of stock", "out-of-stock", "not in stock", "unavailable"]) and any(term in user_question.lower() for term in ["percentage", "%", "proportion", "ratio", "most", "highest", "maximum"]):
                    try:
                        import pandas as pd
                        # Load the CSV directly
                        if csv_file is not None:
                            df = pd.read_csv(csv_file)
                        elif file_path is not None:
                            df = pd.read_csv(file_path)
                        else:
                            raise Exception("No data source available")
                        
                        # Check for availability/stock column with flexible naming
                        stock_column = None
                        for col in df.columns:
                            if col.lower() in ['availability', 'stock', 'status', 'in_stock', 'inventory']:
                                stock_column = col
                                break
                                
                        if stock_column is None:
                            raise Exception("Availability/stock data not found in the dataset")
                            
                        # Get category column with flexible naming
                        category_column = None
                        for col in df.columns:
                            if col.lower() in ['category', 'categories', 'genre', 'type', 'class']:
                                category_column = col
                                break
                                
                        if category_column is None:
                            raise Exception("Category data not found in the dataset")
                        
                        # Calculate percentage of books out of stock per category - handle different stock column formats
                        # Check if 'out of stock' is contained in the stock column value (case insensitive)
                        df['is_out_of_stock'] = df[stock_column].astype(str).str.lower().apply(
                            lambda x: any(term in x for term in ['out of stock', 'out-of-stock', 'unavailable', 'not in stock', '0 in stock'])
                        )
                        
                        category_stats = df.groupby(category_column).apply(
                            lambda x: (x['is_out_of_stock'].sum() / len(x)) * 100 if len(x) > 0 else 0
                        ).reset_index(name='percent_out_of_stock')
                        
                        # Find category with highest percentage
                        if len(category_stats) > 0:
                            # Check if all categories have 0% out of stock books
                            max_percentage = category_stats['percent_out_of_stock'].max()
                            
                            if max_percentage == 0:
                                # All categories have 0% out of stock
                                total_books = len(df)
                                total_categories = len(category_stats)
                                raw_answer = f"None of the categories have any books marked as 'Out of stock'. All {total_books} books across all {total_categories} categories are fully in stock and available for purchase."
                            else:
                                # Sort by percentage out of stock in descending order
                                category_stats = category_stats.sort_values('percent_out_of_stock', ascending=False)
                                top_category = category_stats.iloc[0]
                                cat_name = top_category[category_column]
                                pct = top_category['percent_out_of_stock']
                                
                                # Count books in this category
                                cat_books = len(df[df[category_column] == cat_name])
                                out_of_stock = int(round((pct/100) * cat_books))
                                
                                raw_answer = f"The category with the highest percentage of books marked as 'Out of stock' is {cat_name}, with {pct:.1f}% ({out_of_stock} out of {cat_books} books) marked as 'Out of stock'."
                        else:
                            raw_answer = "No categories found in the dataset."
                            
                        # Create detailed trace for debugging and explanation
                        trace = f"Analyzed CSV data directly to find categories with highest percentage of out-of-stock books:\n"
                        trace += f"Loaded data with {len(df)} books across {df[category_column].nunique()} categories\n"
                        
                        # Add category percentages to trace for transparency
                        trace += "Category statistics (sorted by out-of-stock percentage):\n"
                        sorted_stats = category_stats.sort_values('percent_out_of_stock', ascending=False)
                        for _, row in sorted_stats.iterrows():
                            cat = row[category_column]
                            pct = row['percent_out_of_stock']
                            count = len(df[df[category_column] == cat])
                            out_of_stock_count = int(round((pct/100) * count))
                            trace += f"- {cat}: {pct:.1f}% out of stock ({out_of_stock_count} out of {count} books)\n"
                            
                        # Add explanation for all-zero case
                        if max_percentage == 0:
                            trace += "\nSpecial case detected: All categories have 0% out of stock books\n"
                            trace += "Providing clear message that no books are out of stock rather than singling out a category"
                        
                    except Exception as e:
                        # Fall back to agent if direct approach fails
                        trace = f"Direct approach failed with: {str(e)}\nFalling back to agent..."
                        buf = io.StringIO()
                        # Make sure the question ends with a question mark
                        processed_question = user_question.strip()
                        if not processed_question.endswith("?"):
                            processed_question = processed_question + "?"
                            
                        try:
                            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                                raw_answer = agent.run(processed_question)
                            trace += buf.getvalue()
                        except Exception as e:
                            trace += buf.getvalue()
                            raw_answer = f"I'm sorry, I couldn't analyze which book categories have the highest percentage of out-of-stock books. This seems to be a complex calculation with the current dataset."
                else:
                    # Standard agent approach for other questions
                    # Capture stdout/stderr to get agent traces
                    buf = io.StringIO()
                    
                    # Make sure the question ends with a question mark if it doesn't already
                    processed_question = user_question.strip()
                    if not processed_question.endswith("?"):
                        processed_question = processed_question + "?"
                    
                    try:
                        # Standard agent approach for other questions
                        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                            raw_answer = agent.run(processed_question)
                        trace = buf.getvalue()
                        
                    except Exception as e:
                        # Handle errors gracefully
                        trace = buf.getvalue()
                        error_message = str(e)
                        print(f"Error while running agent: {error_message}")
                        
                        # Generate a helpful error response
                        try:
                            error_llm = OpenAI(temperature=0.7, model_name=MODEL_NAME, request_timeout=30)
                            error_prompt = (
                                "You are a friendly book data assistant. The system couldn't find an answer to the user's question.\n"
                                "Create a polite, helpful response that:\n"
                                "1. Acknowledges we couldn't answer their specific question\n"
                                "2. Suggests how they might rephrase it or ask something different\n"
                                "3. Gives 2-3 example questions they could try instead about books, categories, ratings, or prices\n"
                                "4. Has a warm, helpful tone\n\n"
                                f"USER QUESTION: {user_question}\n\n"
                                "YOUR FRIENDLY RESPONSE:"
                            )
                            buf_error = io.StringIO()
                            with contextlib.redirect_stdout(buf_error), contextlib.redirect_stderr(buf_error):
                                error_response = error_llm(error_prompt)
                            raw_answer = error_response.strip() or "I couldn't find an answer to that question about the books. Could you try rephrasing it or asking something different about the book data?"
                        except Exception:
                            raw_answer = "I couldn't find an answer to that question about the books. Could you try asking about book categories, prices, ratings, or specific titles in our dataset?"
                        buf_error = io.StringIO()
                        with contextlib.redirect_stdout(buf_error), contextlib.redirect_stderr(buf_error):
                            error_response = error_llm(error_prompt)
                        raw_answer = error_response.strip() or "I couldn't find an answer to that question about the books. Could you try rephrasing it or asking something different about the book data?"
                    except Exception:
                        raw_answer = "I couldn't find an answer to that question about the books. Could you try asking about book categories, prices, ratings, or specific titles in our dataset?"
        
        # Process and display the answer
        if process_question:
            # Special post-processing for "out of stock" percentage answers
            if any(phrase in raw_answer.lower() for phrase in ["0% of books marked as \"out of stock\"", "0% out of stock", "0% marked as out of stock", "0 out of", "0% of the books"]):
                # This might be misleading - check if we need to clarify that ALL categories have 0% out of stock
                if not any(phrase in raw_answer.lower() for phrase in ["all categories", "none of the categories", "no categories", "all books across all categories"]):
                    # Create a more accurate message
                    raw_answer = "None of the categories have any books marked as 'Out of stock'. All books across all categories are fully in stock and available for purchase."
            
            # Improve response quality
            improved = raw_answer
            
            # Refine the response if enabled
            if auto_refine:
                try:
                    eval_llm = OpenAI(temperature=0.2, model_name=MODEL_NAME, request_timeout=50)
                    refine_prompt = (
                        "You are given an answer about book data that may need clearer structure and coherence.\n"
                        "Evaluate the answer briefly and produce an improved version.\n"
                        "If the answer contains any technical details like function names, coding terms, or implementation details, "
                        "remove them and replace with book-relevant explanations.\n"
                        "When working with percentages, especially 0% or 100%, provide context to make the answer more informative. For example:\n"
                        "- If all categories have '0% out of stock', the answer should clearly state that NO categories have ANY books marked as out of stock, rather than singling out one category as having the 'highest percentage'\n"
                        "- If a category has '100% out of stock', explain that this means 'no books in this category are currently available'\n"
                        "Return ONLY the improved answer without any references to functions, code syntax, or technical terms.\n\n"
                        "ORIGINAL ANSWER:\n" + raw_answer
                    )
                    buf2 = io.StringIO()
                    with contextlib.redirect_stdout(buf2), contextlib.redirect_stderr(buf2):
                        refine_out = eval_llm(refine_prompt)
                    improved = refine_out.strip()
                except Exception:
                    improved = raw_answer

            # Start with the improved answer
            display_text = improved

            # Rephrase into friendly tone if requested
            if friendly_tone:
                try:
                    reph_llm = OpenAI(temperature=0.7, model_name=MODEL_NAME, request_timeout=30)
                    prompt = (
                        "Rephrase the following answer about book data into a short, friendly, human tone. "
                        "Keep it concise and easy to read. Remove any technical terms, function names, "
                        "or programming concepts that might appear, replacing them with plain language descriptions. "
                        "If the answer contains percentages like '0%' or '100%', provide additional context to make it more informative. "
                        "For example, if a category has '0% out of stock', emphasize this means 'all books are in stock' or 'fully stocked'. "
                        "When dealing with the 'highest percentage of out of stock books' being 0%, make it clear this is actually good news - "
                        "it means no books are out of stock across all categories, or that the category mentioned has no stock issues at all. "
                        "Never mention Python functions, code, or implementation details in your response.\n\n" + improved
                    )
                    buf3 = io.StringIO()
                    with contextlib.redirect_stdout(buf3), contextlib.redirect_stderr(buf3):
                        friendly_answer = reph_llm(prompt)
                    display_text = friendly_answer
                except Exception:
                    display_text = improved

            # Display the answer
            st.markdown("### Answer:")
            st.success(display_text)

            # Show detailed explanation if requested
            if show_details:
                    # Always run the local parser first to extract a structured
                    # expert-style walkthrough from the agent trace (or raw answer).
                    # Then pass that parsed output to the LLM to polish into a
                    # concise, human-expert explanation. If the LLM call fails,
                    # display the parser output so we always have a helpful
                    # walkthrough.
                    parsed_explanation = _parse_trace_to_explanation(trace.strip() or raw_answer, user_question, raw_answer)
                    try:
                        explain_llm = OpenAI(temperature=0.7, model_name=MODEL_NAME, request_timeout=30)
                        explain_prompt = (
                            "You are a friendly teacher explaining data analysis to a student. The following text contains technical steps that were used to analyze data and answer a question.\n"
                            "Transform this into a plain-language explanation as if you're telling a short story. Use a warm, engaging tone like a teacher would use.\n"
                            "Focus on the WHY and WHAT was discovered, not HOW the code works. Include the important numbers and findings but explain their significance in everyday terms.\n"
                            "Avoid technical jargon, code snippets, or programming terms. Instead, use metaphors and simple explanations.\n\n"
                            f"STUDENT'S QUESTION: {user_question}\n\n"
                            "TECHNICAL STEPS TAKEN:\n" + parsed_explanation + "\n\n"
                            "YOUR FRIENDLY EXPLANATION (use paragraphs, not numbered points):\n"
                        )
                        buf4 = io.StringIO()
                        with contextlib.redirect_stdout(buf4), contextlib.redirect_stderr(buf4):
                            explanation = explain_llm(explain_prompt)
                        # If the LLM returned an empty string for some reason,
                        # fall back to the parser output.
                        if not explanation or not explanation.strip():
                            explanation = parsed_explanation
                    except Exception:
                        # LLM failed — show the parser output so users still
                        # get a clear, expert-style walkthrough.
                        explanation = parsed_explanation

                    with st.expander("📊 How I found your answer"):
                        if not trace:  # This is a greeting or general help question with no trace
                            if any(pattern in user_question.lower() for pattern in ['how can you help', 'what can you do', 'help me', 'what do you do']):
                                st.markdown("I can answer various questions about your book dataset, such as:")
                                st.markdown("• Finding books in specific categories")
                                st.markdown("• Identifying the highest or lowest rated books")
                                st.markdown("• Analyzing price ranges or averages")
                                st.markdown("• Checking availability of certain titles")
                                st.markdown("• Comparing different book attributes")
                            else:
                                st.markdown("I'm your books information assistant, ready to help you analyze your book data. Feel free to ask me specific questions about your books dataset.")
                        elif explanation.strip() == "No parsed steps found in trace.":
                            st.markdown("I analyzed your question directly using the book data available. If you'd like more specific details, please ask a more targeted question about the books in your dataset.")
                            # Don't show raw trace or technical details
                        else:
                            # Add a friendly header to the explanation
                            st.markdown("### Here's how I found your answer:")
                            st.markdown(explanation)


warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
warnings.filterwarnings("ignore", message="You are trying to use a chat model. This way of initializing it is no longer supported")

if __name__ == "__main__":
    
    if os.getenv("RUNNING_VIA_STREAMLIT") != "1":
        env = os.environ.copy()
        env["RUNNING_VIA_STREAMLIT"] = "1"
        # Use the Python module entry so we don't rely on a separate 'streamlit' binary.
        cmd = [sys.executable, "-m", "streamlit", "run", str(Path(__file__).name)]
        print("Launching Streamlit...")
        subprocess.run(cmd, env=env)
        sys.exit(0)
    main()
