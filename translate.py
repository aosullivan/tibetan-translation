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

# Change the output file name to 'translation'
output_file_path = 'translation'
if os.path.exists(output_file_path):
    os.remove(output_file_path)
    logging.info(f"Deleted existing file: {output_file_path}")

def get_claude_response(prompt):
    logger.info("Initializing the client with API key")
    
    # Initialize the client with your API key
    client = Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
    
    logger.info("Reading the content of the file")
    # Read the content of the file
    with open('sdom-gsum-tibetan.txt', 'r', encoding='utf-8') as file:
        content = file.read()
    
    logger.info("Splitting the content into 250 parts")
    # Split the content into 250 parts
    parts = [content[i:i + len(content) // 250] for i in range(0, len(content), len(content) // 250)]
    
    # Limit the number of chunks to 3
    # parts = parts[:3]
    
    for i, part in enumerate(parts):
        logger.info(f"Processing part {i+1}/{len(parts)}")
        
        # Send message to Claude
        full_response = translate(client, part)
        
        # Write the response to the output file as each part is processed
        with open(output_file_path, 'a', encoding='utf-8') as output_file:
            output_file.write(full_response + "\n")
        
        logger.info(f"Finished processing part {i+1}/{len(parts)}")

def translate(client, part):
    logger = logging.getLogger(__name__)
    max_retries = 5
    initial_delay = 1

    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model="claude-3-5-sonnet-latest",
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": f"""
                     You are an AI agent for translating classical Tibetan texts into English. 
                     I will send you contiguous parts of a text one at a time and you should translate these into English. 
                     Make sure the whole Tibetan text is translated and represented in English and do not leave anything out. 
                     You can use sanskrit terms and include the original Tibetan terms in brackets if it add clarity. 
                     Give me the English text with your full and complete translation.
                     Use enumerations where possible, e.g. if the text says 'there are two parts', create a numerical list for the two parts and a heading for each part.
                     e.g. 'Second, regarding the definitive meaning:' should be translated as '2. Regarding the Definitive Meaning'
                     e.g. 'The forty-first question is: Where is it explained that Rigden Dragpo is actually an emanation of Vajrapani?' should be translated as, 'Question 41. Where is it explained that Rigden Dragpo is actually an emanation of Vajrapani?'
                     Do not add additional comments or questions of your own,  but only return the translation with no other words at all. 
                     Here is the text:\n\n{part}
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

def main():
    logger.info("Loading environment variables")
    load_dotenv()
    
    # Make sure you have your API key set as an environment variable
    if not os.environ.get('ANTHROPIC_API_KEY'):
        logger.error("Please set your ANTHROPIC_API_KEY environment variable")
        return
    
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