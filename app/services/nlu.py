import re
import asyncio
import time
from typing import Dict, Any, Optional, List, Tuple
try:
    import openai
except ImportError:
    openai = None
from app.core.config import settings
from app.core.logger import logger
from app.models.candidate import ShiftPreference, LanguageSkill


class NLUService:
    """Natural Language Understanding service for Hinglish entity extraction."""
    
    def __init__(self):
        if settings.openai_api_key:
            openai.api_key = settings.openai_api_key
    
    async def extract_entities(self, text: str, field: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Extract entities from Hinglish text for a specific field.
        
        Args:
            text (str): User input text
            field (str): Field being collected (pincode, salary, etc.)
            context (Dict[str, Any]): Conversation context
            
        Returns:
            Dict[str, Any]: Extracted entities and metadata
        """
        start_time = time.time()
        
        # Normalize text first
        normalized_text = self._normalize_text(text)
        
        # Use appropriate extraction method based on field
        if field == "pincode":
            result = await self._extract_pincode(normalized_text)
        elif field == "availability_date":
            result = await self._extract_availability(normalized_text)
        elif field == "preferred_shift":
            result = await self._extract_shift(normalized_text)
        elif field == "expected_salary":
            result = await self._extract_salary(normalized_text)
        elif field == "languages":
            result = await self._extract_languages(normalized_text)
        elif field == "has_two_wheeler":
            result = await self._extract_boolean(normalized_text, field)
        elif field == "total_experience_months":
            result = await self._extract_experience(normalized_text)
        else:
            # Generic extraction using OpenAI
            result = await self._extract_with_openai(normalized_text, field, context)
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.bind(metrics=True).info({
            "event": "nlu_extraction",
            "field": field,
            "input_text": text,
            "extracted_entities": result,
            "processing_time_ms": processing_time
        })
        
        return result
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for entity extraction."""
        if not text:
            return text
            
        # Convert to lowercase
        text = text.lower().strip()
        
        # Common Hindi to English normalizations
        normalizations = {
            # Numbers
            "ek": "1", "do": "2", "teen": "3", "char": "4", "panch": "5",
            "che": "6", "saat": "7", "aath": "8", "nau": "9", "das": "10",
            "gyarah": "11", "barah": "12", "terah": "13", "chaudah": "14", "pandrah": "15",
            "solah": "16", "satrah": "17", "atharah": "18", "unnis": "19", "bees": "20",
            "hazaar": "thousand", "lakh": "lakh",
            
            # Time/Date
            "aaj": "today", "kal": "tomorrow", "parso": "day after tomorrow",
            "abhi": "now", "turant": "immediately", "jaldi": "soon",
            
            # Boolean
            "haan": "yes", "han": "yes", "ji haan": "yes", "bilkul": "yes",
            "nahi": "no", "nahin": "no", "na": "no", "bilkul nahi": "no",
            
            # Experience mishear fixes
            "saal": "years", "sal": "years",
            "yer": "years", "yers": "years", "yr": "years", "yrs": "years", "y": "years",
            "ear": "years", "ears": "years", "yar": "years", "yars": "years",
            "mahina": "months", "mahine": "months",
            "mnth": "months", "mnths": "months", "mth": "months", "mths": "months",
            "mon": "months", "mons": "months"
        }
        
        for hindi, english in normalizations.items():
            text = re.sub(rf"\b{re.escape(hindi)}\b", english, text)
        
        return text
    
    async def _extract_pincode(self, text: str) -> Dict[str, Any]:
        """Extract pincode from text."""
        # Look for 6-digit numbers
        pincode_pattern = r'\b\d{6}\b'
        matches = re.findall(pincode_pattern, text)
        
        if matches:
            return {
                "value": matches[0],
                "confidence": 0.9,
                "field": "pincode",
                "method": "regex"
            }
        
        # Try to extract from OpenAI if no direct match
        return await self._extract_with_openai(text, "pincode")
    
    async def _extract_availability(self, text: str) -> Dict[str, Any]:
        """Extract availability date from text."""
        # Check for common patterns
        if any(word in text for word in ["today", "aaj"]):
            return {"value": "today", "confidence": 0.9, "field": "availability_date", "method": "rule"}
        elif any(word in text for word in ["tomorrow", "kal"]):
            return {"value": "tomorrow", "confidence": 0.9, "field": "availability_date", "method": "rule"}
        elif any(word in text for word in ["day after tomorrow", "parso"]):
            return {"value": "day after tomorrow", "confidence": 0.9, "field": "availability_date", "method": "rule"}
        elif any(word in text for word in ["immediately", "turant", "abhi", "now"]):
            return {"value": "immediately", "confidence": 0.9, "field": "availability_date", "method": "rule"}
        elif any(word in text for word in ["soon", "jaldi"]):
            return {"value": "soon", "confidence": 0.8, "field": "availability_date", "method": "rule"}
        
        # Look for date patterns (dd/mm/yyyy, dd-mm-yyyy)
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
        matches = re.findall(date_pattern, text)
        if matches:
            return {"value": matches[0], "confidence": 0.8, "field": "availability_date", "method": "regex"}
        
        return await self._extract_with_openai(text, "availability_date")
    
    async def _extract_shift(self, text: str) -> Dict[str, Any]:
        """Extract shift preference from text."""
        shift_keywords = {
            "morning": ["morning", "subah", "savera"],
            "afternoon": ["afternoon", "dopahar", "dupa"],
            "evening": ["evening", "shaam", "sham"],
            "night": ["night", "raat", "rat"],
            "flexible": ["flexible", "koi bhi", "any", "anytime"]
        }
        
        for shift, keywords in shift_keywords.items():
            if any(keyword in text for keyword in keywords):
                return {
                    "value": shift,
                    "confidence": 0.9,
                    "field": "preferred_shift",
                    "method": "rule"
                }
        
        return await self._extract_with_openai(text, "shift_preference")
    
    async def _extract_salary(self, text: str) -> Dict[str, Any]:
        """Extract salary amount from text."""
        # Look for numbers followed by salary keywords
        salary_patterns = [
            r'(\d+)\s*(?:thousand|hazaar|k)',
            r'(\d+)\s*(?:lakh|lac)',
            r'(\d+)\s*(?:rupees|rupaye|rs)',
            r'\b(\d{4,6})\b'  # 4-6 digit numbers
        ]
        
        for pattern in salary_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                amount = int(matches[0])
                
                # Convert thousands and lakhs
                if any(word in text.lower() for word in ['thousand', 'hazaar', 'k']):
                    amount *= 1000
                elif any(word in text.lower() for word in ['lakh', 'lac']):
                    amount *= 100000
                
                return {
                    "value": amount,
                    "confidence": 0.9,
                    "field": "expected_salary",
                    "method": "regex"
                }
        
        return await self._extract_with_openai(text, "salary")
    
    async def _extract_languages(self, text: str) -> Dict[str, Any]:
        """Extract language skills from text."""
        language_keywords = {
            "hindi": ["hindi", "हिंदी"],
            "english": ["english", "angrezi"],
            "marathi": ["marathi", "मराठी"],
            "bengali": ["bengali", "bangla", "বাংলা"],
            "tamil": ["tamil", "தமிழ்"],
            "telugu": ["telugu", "తెలుగు"],
            "gujarati": ["gujarati", "ગુજરાતી"],
            "kannada": ["kannada", "ಕನ್ನಡ"],
            "punjabi": ["punjabi", "ਪੰਜਾਬੀ"]
        }
        
        found_languages = []
        for lang, keywords in language_keywords.items():
            if any(keyword in text.lower() for keyword in keywords):
                found_languages.append(lang)
        
        if found_languages:
            return {
                "value": found_languages,
                "confidence": 0.9,
                "field": "languages",
                "method": "rule"
            }
        
        return await self._extract_with_openai(text, "languages")
    
    async def _extract_boolean(self, text: str, field: str) -> Dict[str, Any]:
        """Extract yes/no response from text."""
        yes_keywords = ["yes", "haan", "han", "ji", "bilkul", "है", "h"]
        no_keywords = ["no", "nahi", "nahin", "na", "नहीं", "n"]
        
        if any(keyword in text.lower() for keyword in yes_keywords):
            return {"value": True, "confidence": 0.9, "field": field, "method": "rule"}
        elif any(keyword in text.lower() for keyword in no_keywords):
            return {"value": False, "confidence": 0.9, "field": field, "method": "rule"}
        
        return await self._extract_with_openai(text, field)
    
    async def _extract_experience(self, text: str) -> Dict[str, Any]:
        """Extract work experience in months."""
        # Look for patterns like "2 years", "6 months", with common ASR mishears
        experience_patterns = [
            r'(\d+\.?\d*)\s*(?:year|years|yr|yrs|y)',
            r'(\d+\.?\d*)\s*(?:month|months|mnth|mnths|mth|mths|mon|mons|mahina|mahine)'
        ]
        
        total_months = 0
        
        for pattern in experience_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                value = float(matches[0])
                
                if any(word in text.lower() for word in ['year', 'years', 'yr', 'yrs', 'y']):
                    total_months += int(round(value * 12))
                else:  # months variants
                    total_months += int(round(value))
        
        if total_months > 0:
            return {
                "value": total_months,
                "confidence": 0.9,
                "field": "total_experience_months",
                "method": "regex"
            }
        
        # Check for "fresher" or "no experience"
        if any(word in text.lower() for word in ['fresher', 'fresh', 'naya', 'no experience', 'koi experience nahi']):
            return {
                "value": 0,
                "confidence": 0.9,
                "field": "total_experience_months",
                "method": "rule"
            }
        
        # Numeric-only fallback: assume small numbers (<=10) are years, else months
        bare_num = re.search(r'\b(\d{1,3}(?:\.\d+)?)\b', text)
        if bare_num:
            val = float(bare_num.group(1))
            if val <= 10:
                months = int(round(val * 12))
            else:
                months = int(round(val))
            return {
                "value": months,
                "confidence": 0.6,
                "field": "total_experience_months",
                "method": "fallback"
            }
        
        return await self._extract_with_openai(text, "experience")
    
    async def _extract_with_openai(self, text: str, field: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Use OpenAI for complex entity extraction."""
        if not settings.openai_api_key:
            return {"value": None, "confidence": 0.0, "field": field, "method": "none"}
        
        try:
            prompt = self._create_extraction_prompt(text, field, context)
            
            response = await asyncio.to_thread(
                openai.chat.completions.create,
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse the response
            return self._parse_openai_response(result_text, field)
            
        except Exception as e:
            logger.error(f"OpenAI extraction failed: {e}")
            return {"value": None, "confidence": 0.0, "field": field, "method": "failed"}
    
    def _create_extraction_prompt(self, text: str, field: str, context: Dict[str, Any] = None) -> str:
        """Create prompt for OpenAI entity extraction."""
        prompts = {
            "pincode": f"Extract the 6-digit pincode from this Hinglish text: '{text}'. Reply with just the pincode number or 'None' if not found.",
            "salary": f"Extract the monthly salary amount in INR from this Hinglish text: '{text}'. Reply with just the number or 'None' if not found.",
            "availability_date": f"Extract when the person can start work from this Hinglish text: '{text}'. Reply with 'today', 'tomorrow', 'soon', or a specific date.",
            "shift_preference": f"Extract work shift preference from this Hinglish text: '{text}'. Reply with 'morning', 'afternoon', 'evening', 'night', or 'flexible'.",
            "languages": f"Extract languages the person can speak from this Hinglish text: '{text}'. Reply with language names separated by commas.",
            "experience": f"Extract work experience in months from this Hinglish text: '{text}'. Reply with just the number of months or '0' for fresher."
        }
        
        return prompts.get(field, f"Extract relevant information for {field} from: '{text}'")
    
    def _parse_openai_response(self, response: str, field: str) -> Dict[str, Any]:
        """Parse OpenAI response into structured format."""
        response = response.strip().lower()
        
        if response == "none" or not response:
            return {"value": None, "confidence": 0.0, "field": field, "method": "openai"}
        
        # Field-specific parsing
        if field == "expected_salary":
            try:
                value = int(re.search(r'\d+', response).group())
                return {"value": value, "confidence": 0.7, "field": field, "method": "openai"}
            except:
                return {"value": None, "confidence": 0.0, "field": field, "method": "openai"}
        
        elif field == "has_two_wheeler":
            value = response in ["yes", "true", "1", "haan"]
            return {"value": value, "confidence": 0.7, "field": field, "method": "openai"}
        
        elif field == "total_experience_months":
            try:
                value = int(re.search(r'\d+', response).group())
                return {"value": value, "confidence": 0.7, "field": field, "method": "openai"}
            except:
                return {"value": None, "confidence": 0.0, "field": field, "method": "openai"}
        
        else:
            return {"value": response, "confidence": 0.7, "field": field, "method": "openai"} 