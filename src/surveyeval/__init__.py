#  Copyright (c) 2024 Higher Bar AI, PBC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from .evaluation_engine import EvaluationEngine, EvaluationLens
from .core_evaluation_lenses import (BiasEvaluationLens, PhrasingEvaluationLens, TranslationEvaluationLens,
                                     ValidatedInstrumentEvaluationLens)
try:
    from .survey_parser import SurveyInterface
except ImportError as e:
    # we expect an import error if the optional dependencies are not installed
    SurveyInterface = None
    print(f"FYI: surveyeval.survey_parser module not loaded; if you want to parse survey documents, you'll need to "
          f"pip install surveyeval[parser] to install the necessary dependencies. Import exception: {e}")

# report our current version, as installed
from importlib.metadata import version
try:
    __version__ = version("surveyeval")
except Exception:
    # (ignore exceptions when developing and testing locally)
    pass
