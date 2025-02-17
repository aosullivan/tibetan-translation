import anthropic
import os
from anthropic import Anthropic
from dotenv import load_dotenv
import logging
import traceback
import time
import random
from anthropic import Anthropic, APIError, RateLimitError

# Configure logging at the top of the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

output_file_path = 'translation.txt'
if os.path.exists(output_file_path):
    os.remove(output_file_path)
    logging.info(f"Deleted existing file: {output_file_path}")

# Initialize the client with your API key
client = None

def get_claude_response(prompt):
    logger.info("Reading the content of the file")
    with open('sdom-gsum-tibetan.txt', 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Change to chunk by character count instead of fixed number of parts
    chunk_size = 10000
    logger.info(f"Splitting the content into chunks of approximately {chunk_size} characters")
    
    # Create chunks of approximately 10k characters, trying to break at newlines
    parts = []
    start = 0
    while start < len(content):
        end = start + chunk_size
        
        # If we're not at the end of the content, try to find a natural breaking point
        if end < len(content):
            # Look for the last newline within the chunk size
            last_newline = content.rfind('\n', start, end)
            if last_newline != -1:
                end = last_newline + 1
        else:
            end = len(content)
            
        parts.append(content[start:end])
        start = end
    
    logger.info(f"Split content into {len(parts)} chunks")
    
    previous_translation = ""
    translation_summary = ""
    
    for i, part in enumerate(parts):
        logger.info(f"Processing part {i+1}/{len(parts)}")
        
        # Get translation with context
        full_response = translate(client, part, previous_translation, translation_summary)
        
        # Update context for next iteration
        previous_translation = full_response
        
        # Every 5 chunks, update the summary to prevent context window from growing too large
        if i % 5 == 0:
            translation_summary = get_summary(client, previous_translation)
        
        # Write the response to the output file as each part is processed
        with open(output_file_path, 'a', encoding='utf-8') as output_file:
            output_file.write(full_response + "\n")
        
        logger.info(f"Finished processing part {i+1}/{len(parts)}")

def translate(client, part, previous_translation, translation_summary):
    logger = logging.getLogger(__name__)
    max_retries = 5
    initial_delay = 1

    for attempt in range(max_retries):
        try:
            context_prompt = ""
            if translation_summary:
                context_prompt = f"Summary of previous content: {translation_summary}\n\n"
            if previous_translation:
                context_prompt += f"Previous chunk's translation: {previous_translation}\n\n"

            message = client.messages.create(
                model="claude-3-5-sonnet-latest",
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": f"""
                     You are an AI agent for translating classical Tibetan texts into English. 
                     You will receive parts of a text sequentially. Your task is to translate the current part 
                     while maintaining consistency with the previous translations.

                     {context_prompt}
                     
                     Guidelines:
                     - Maintain consistency in terminology with previous translations
                     - Ensure smooth transitions between chunks
                     - Use Sanskrit terms if that is appropriate according to the norms of Tibetan translations into English
                     - Include original Tibetan terms in brackets if the term is particularly technical or obscure, or the translation needs clarification
                     - Use enumerations where applicable (e.g., "Second, blah blah" becomes "2. Blah blah")
                     - If the text says something like 'There are two parts', list them as '1.' and '2.' Remember these enumerations and use them as headings of they are explained in more detail later
                     - If an enumerated part has subparts or subsections, enumerate them as 1.1, 1.2, etc. and likewise if there are further sub-enumerations then use 1.1.1, 1.1.2 and so forth 
                     - Do not put a dot after subenumerations, e.g. 1.1.2 not 1.1.2. and 1.2 not 1.2. and so forth
                     - Create numbered lists for any enumerated items, but otherwise do not use lists
                     - Do not say 'Here is the translation' or add any of your own comments or words. Give me ONLY the translated words.
                     
                     Translate the following text, ensuring it flows naturally from the previous content:

                     {part}
                     """}
                ]
            )
            
            # Concatenate all text content
            return ''.join(block.text for block in message.content)

        except RateLimitError as e:
            if attempt == max_retries - 1:
                logger.error(f"Rate limit exceeded after {max_retries} attempts")
                raise
            
            # Get retry delay from response headers if available
            retry_after = int(e.response.headers.get('retry-after', initial_delay))
            
            # Add jitter to prevent thundering herd
            sleep_time = retry_after + random.uniform(0, 0.5)
            logger.warning(f"Rate limited. Retrying in {sleep_time:.1f} seconds...")
            time.sleep(sleep_time)
            
            # Exponential backoff
            initial_delay *= 2
            
        except APIError as e:
            logger.error(f"API error: {str(e)}")
            if e.status_code and e.status_code >= 500:
                if attempt == max_retries - 1:
                    raise
                time.sleep(initial_delay * (2 ** attempt))
            else:
                # Don't retry other API errors
                raise

def get_summary(client, text):
    """Generate a brief summary of the translated text to maintain context"""
    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=500,
            messages=[
                {"role": "user", "content": f"""
                 Provide a brief summary of the key points and context from this translated text. 
                 Focus on main themes, key terms, and any ongoing topics that would be important 
                 for maintaining consistency in subsequent translations:

                 {text}
                 """}
            ]
        )
        return ''.join(block.text for block in message.content)
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return ""

def main():
    global client
    logger.info("Loading environment variables")
    load_dotenv()
    
    # Make sure you have your API key set as an environment variable
    if not os.environ.get('ANTHROPIC_API_KEY'):
        logger.error("Please set your ANTHROPIC_API_KEY environment variable")
        return
    
    logger.info("Initializing the client with API key")
    client = Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
    
    # Example prompt
    prompt = "What are three interesting facts about penguins?"
    
    try:
        logger.info("Starting the Claude response process")
        response = get_claude_response(prompt)
        logger.info("Claude's response received")
        print("Claude's response:")
        print(response)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()