curl https://api.anthropic.com/v1/messages \
        --header "x-api-key: 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" \
        --header "anthropic-version: 2023-06-01" \
        --header "content-type: application/json" \
        --data \
    '{
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "Hello, world"}
        ]
    }'