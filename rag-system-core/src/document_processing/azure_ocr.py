import os
from dotenv import load_dotenv
load_dotenv()
import requests
import time

AZURE_ENDPOINT = os.getenv("COMPUTER_VISION_ENDPOINT")
AZURE_KEY = os.getenv("COMPUTER_VISION_KEY")
AZURE_OCR_MAX_RETRIES = int(os.getenv("AZURE_OCR_MAX_RETRIES", 3))


def azure_ocr_image(image_bytes, max_retries=None):
    url = AZURE_ENDPOINT.rstrip("/") + "/vision/v3.2/read/analyze"
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_KEY,
        "Content-Type": "application/octet-stream"
    }
    retries = max_retries if max_retries is not None else AZURE_OCR_MAX_RETRIES
    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, data=image_bytes)
            response.raise_for_status()
            operation_url = response.headers["Operation-Location"]
            # Poll for result
            for _ in range(30):
                result = requests.get(operation_url, headers={"Ocp-Apim-Subscription-Key": AZURE_KEY})
                result_json = result.json()
                if result_json.get("status") in ["succeeded", "failed"]:
                    break
                time.sleep(1)
            if result_json.get("status") == "succeeded":
                lines = []
                for read_result in result_json["analyzeResult"]["readResults"]:
                    for line in read_result["lines"]:
                        lines.append(line["text"])
                return "\n".join(lines)
            elif result_json.get("status") == "failed":
                raise RuntimeError("Azure OCR analysis failed")
            else:
                raise RuntimeError(f"Azure OCR unknown status: {result_json}")
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 429 and attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            raise
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError(f"Azure OCR failed after {retries} attempts") 