# This file defines any custom types or data structures used throughout the project.
# It may include type aliases or interfaces for better type checking.

from typing import Any, Dict, List, Tuple

# Type alias for the dataset
Dataset = Tuple[List[Dict[str, Any]], List[int]]

# Type alias for model parameters
ModelParams = Dict[str, Any]

# Type alias for cross-validation results
CVResults = Dict[str, float]