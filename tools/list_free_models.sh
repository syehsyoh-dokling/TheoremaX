#!/usr/bin/env bash
# Catalogues of candidate models: NVIDIA NIM (free tier with our key), OpenRouter :free, NagaAI :free.
H=/home/webmaster
curl -s -m 60 -H "Authorization: Bearer $(cat $H/.agenthon_nvidia_key)" https://integrate.api.nvidia.com/v1/models > /tmp/m_nv.json
curl -s -m 60 https://openrouter.ai/api/v1/models > /tmp/m_or.json
curl -s -m 60 -H "Authorization: Bearer $(cat $H/.agenthon_naga_key)" https://api.naga.ac/v1/models > /tmp/m_ng.json
python3 -c "
import json,re
pat=re.compile(r'(nemotron|qwen3|deepseek|gpt-oss|kimi|glm|llama-4|mistral-large|devstral|minimax|ultra|coder)',re.I)
nv=[m['id'] for m in json.load(open('/tmp/m_nv.json')).get('data',[])]
print('NVIDIA', len(nv)); print('  ', ' '.join(sorted(x for x in nv if pat.search(x))))
o=json.load(open('/tmp/m_or.json')).get('data',[])
f=[m for m in o if m['id'].endswith(':free')]
print('OPENROUTER free', len(f))
for m in f: print('   ', m['id'], m.get('context_length'), 'tools' if 'tools' in (m.get('supported_parameters') or []) else '')
ng=[m['id'] for m in json.load(open('/tmp/m_ng.json')).get('data',[]) if m['id'].endswith(':free')]
print('NAGA free', ' '.join(ng))
"
rm -f /tmp/m_nv.json /tmp/m_or.json /tmp/m_ng.json
