"""
Scipy utility module for statistical operations.

This module provides a centralized import and utility functions for scipy,
ensuring consistent statistical calculations across the codebase.
All statistical operations requiring scipy should use this module.
"""

from typing import List, Tuple, Union

import structlog

logger = structlog.get_logger(__name__)

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
    logger.info("Scipy successfully imported for statistical operations")
except ImportError as e:
    SCIPY_AVAILABLE = False
    logger.error(
        "Scipy is required for statistical operations but is not available. "
        "Please install scipy: pip install scipy>=1.11.0",
        error=str(e)
    )
    raise ImportError(
        "Scipy is required for statistical operations. "
        "Please install with: pip install scipy>=1.11.0"
    ) from e


def perform_t_test(
    sample_a: List[float], 
    sample_b: List[float],
    equal_var: bool = True
) -> Tuple[float, float]:
    """
    Perform independent t-test between two samples.
    
    Args:
        sample_a: First sample data
        sample_b: Second sample data  
        equal_var: Whether to assume equal variances (Welch's t-test if False)
        
    Returns:
        Tuple of (t_statistic, p_value)
        
    Raises:
        ImportError: If scipy is not available
        ValueError: If samples are empty or invalid
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("Scipy is required for t-test operations")
        
    if not sample_a or not sample_b:
        raise ValueError("Both samples must contain data")
        
    if len(sample_a) < 2 or len(sample_b) < 2:
        raise ValueError("Each sample must contain at least 2 data points")
        
    t_stat, p_value = stats.ttest_ind(sample_a, sample_b, equal_var=equal_var)
    
    logger.debug(
        "T-test completed",
        sample_a_size=len(sample_a),
        sample_b_size=len(sample_b),
        t_statistic=float(t_stat),
        p_value=float(p_value),
        equal_var=equal_var
    )
    
    return float(t_stat), float(p_value)


def perform_one_way_anova(*sample_groups: List[float]) -> Tuple[float, float]:
    """
    Perform one-way ANOVA across multiple sample groups.
    
    Args:
        *sample_groups: Variable number of sample groups
        
    Returns:
        Tuple of (f_statistic, p_value)
        
    Raises:
        ImportError: If scipy is not available
        ValueError: If insufficient groups or samples
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("Scipy is required for ANOVA operations")
        
    if len(sample_groups) < 2:
        raise ValueError("ANOVA requires at least 2 sample groups")
        
    # Validate each group has sufficient data
    for i, group in enumerate(sample_groups):
        if not group or len(group) < 2:
            raise ValueError(f"Sample group {i} must contain at least 2 data points")
    
    f_stat, p_value = stats.f_oneway(*sample_groups)
    
    logger.debug(
        "One-way ANOVA completed",
        num_groups=len(sample_groups),
        group_sizes=[len(group) for group in sample_groups],
        f_statistic=float(f_stat),
        p_value=float(p_value)
    )
    
    return float(f_stat), float(p_value)


def calculate_t_critical(
    degrees_of_freedom: int, 
    alpha: float = 0.05,
    two_tailed: bool = True
) -> float:
    """
    Calculate critical t-value for given degrees of freedom and significance level.
    
    Args:
        degrees_of_freedom: Degrees of freedom
        alpha: Significance level (default 0.05)
        two_tailed: Whether this is a two-tailed test
        
    Returns:
        Critical t-value
        
    Raises:
        ImportError: If scipy is not available
        ValueError: If parameters are invalid
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("Scipy is required for t-distribution operations")
        
    if degrees_of_freedom <= 0:
        raise ValueError("Degrees of freedom must be positive")
        
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1")
    
    # For two-tailed test, divide alpha by 2
    tail_alpha = alpha / 2 if two_tailed else alpha
    
    # Get critical value (upper tail)
    t_critical = stats.t.ppf(1 - tail_alpha, degrees_of_freedom)
    
    logger.debug(
        "T-critical calculated",
        degrees_of_freedom=degrees_of_freedom,
        alpha=alpha,
        two_tailed=two_tailed,
        t_critical=float(t_critical)
    )
    
    return float(t_critical)


def calculate_p_value_from_t_stat(
    t_statistic: float, 
    degrees_of_freedom: int,
    two_tailed: bool = True
) -> float:
    """
    Calculate p-value from t-statistic and degrees of freedom.
    
    Args:
        t_statistic: The t-statistic value
        degrees_of_freedom: Degrees of freedom
        two_tailed: Whether this is a two-tailed test
        
    Returns:
        P-value
        
    Raises:
        ImportError: If scipy is not available
        ValueError: If parameters are invalid
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("Scipy is required for p-value calculations")
        
    if degrees_of_freedom <= 0:
        raise ValueError("Degrees of freedom must be positive")
    
    # Calculate p-value using survival function (1 - CDF)
    if two_tailed:
        p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), degrees_of_freedom))
    else:
        p_value = 1 - stats.t.cdf(t_statistic, degrees_of_freedom)
    
    logger.debug(
        "P-value calculated from t-statistic",
        t_statistic=float(t_statistic),
        degrees_of_freedom=degrees_of_freedom,
        two_tailed=two_tailed,
        p_value=float(p_value)
    )
    
    return float(p_value)


def get_scipy_version() -> str:
    """Get the version of scipy being used."""
    if not SCIPY_AVAILABLE:
        return "Not available"
    
    try:
        import scipy
        return scipy.__version__
    except AttributeError:
        return "Unknown"