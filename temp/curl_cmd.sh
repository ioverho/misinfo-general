curl https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent \
  -H "x-goog-api-key: AIzaSyA-kh5ZcPrGq86xmKHWx_rOhx2r7LKRfDQ" \
  -H 'Content-Type: application/json' \
  -X POST \
  -d '{
    "contents": [
      {
        "parts": [
          {
            "text": "In a few words, how does AI work?",
          }
        ]
      }
    ],
    "generationConfig": {
      "temperature": 0.0,
      "thinkingConfig": {
        "thinkingBudget": 1024,
        "includeThoughts": true,
      }
    }
  }'