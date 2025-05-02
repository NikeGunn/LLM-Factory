#!/bin/bash
if [ "$CI" != "true" ]; then
  pip install -r requirements.txt
  python -m spacy download en_core_web_sm
else
  pip install -r requirements-ci.txt
fi

#!/bin/bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
