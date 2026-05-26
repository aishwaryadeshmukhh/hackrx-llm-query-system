"""
Module: performance_monitor.py
Functionality: Performance monitoring utilities.
"""
import time
import os
from typing import Dict, Any
from contextlib import contextmanager

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

class PerformanceMonitor:
    """Monitor system performance during indexing operations."""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = None
        self.psutil_available = PSUTIL_AVAILABLE
        
        # Only initialize psutil process if available
        if self.psutil_available:
            try:
                self.process = psutil.Process(os.getpid())
            except Exception:
                self.process = None
                self.psutil_available = False
        else:
            self.process = None
    
    @contextmanager
    def monitor_operation(self, operation_name: str):
        """Context manager to monitor a specific operation."""
        self.start_operation(operation_name)
        try:
            yield self
        finally:
            self.end_operation(operation_name)
    
    def start_operation(self, operation_name: str):
        """Start monitoring an operation."""
        self.start_time = time.time()
        
        # Get initial memory if psutil is available
        if self.psutil_available and self.process:
            initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        else:
            initial_memory = 0.0
        
        self.metrics[operation_name] = {
            'start_time': self.start_time,
            'initial_memory_mb': initial_memory,
            'peak_memory_mb': initial_memory,
            'cpu_percent': []
        }
    
    def end_operation(self, operation_name: str):
        """End monitoring an operation."""
        if operation_name not in self.metrics:
            return
        
        end_time = time.time()
        
        # Get final memory if psutil is available
        if self.psutil_available and self.process:
            final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        else:
            final_memory = 0.0
        
        self.metrics[operation_name].update({
            'end_time': end_time,
            'duration_seconds': end_time - self.metrics[operation_name]['start_time'],
            'final_memory_mb': final_memory,
            'memory_increase_mb': final_memory - self.metrics[operation_name]['initial_memory_mb']
        })
    
    def update_memory_peak(self, operation_name: str):
        """Update peak memory usage for an operation."""
        if operation_name in self.metrics and self.psutil_available and self.process:
            current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            self.metrics[operation_name]['peak_memory_mb'] = max(
                self.metrics[operation_name]['peak_memory_mb'], 
                current_memory
            )
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get current system information."""
        if not self.psutil_available:
            return {
                'cpu_percent': 0.0,
                'memory_percent': 0.0,
                'available_memory_gb': 0.0,
                'psutil_available': False,
                'message': 'psutil not available - install with: pip install psutil'
            }
        
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'available_memory_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024,
                'psutil_available': True
            }
        except Exception as e:
            return {
                'cpu_percent': 0.0,
                'memory_percent': 0.0,
                'available_memory_gb': 0.0,
                'psutil_available': False,
                'error': str(e)
            }
    
    def get_recommendations(self, operation_name: str) -> list:
        """Get performance recommendations based on metrics."""
        if operation_name not in self.metrics:
            return []
        
        recommendations = []
        metrics = self.metrics[operation_name]
        
        # Check duration
        if metrics.get('duration_seconds', 0) > 60:
            recommendations.append("⏰ Indexing took over 1 minute. Consider reducing chunk size or document count.")
        
        # Check memory usage
        if metrics.get('memory_increase_mb', 0) > 500:
            recommendations.append("🧠 High memory usage detected. Consider processing documents in smaller batches.")
        
        # Check system resources
        system_info = self.get_system_info()
        if system_info['memory_percent'] > 80:
            recommendations.append("💾 System memory usage is high. Close other applications to improve performance.")
        
        if system_info['cpu_percent'] > 80:
            recommendations.append("🔥 CPU usage is high. Consider reducing batch size for embedding generation.")
        
        return recommendations
    
    def format_metrics(self, operation_name: str) -> str:
        """Format metrics for display."""
        if operation_name not in self.metrics:
            return "No metrics available"
        
        metrics = self.metrics[operation_name]
        duration = metrics.get('duration_seconds', 0)
        memory_increase = metrics.get('memory_increase_mb', 0)
        peak_memory = metrics.get('peak_memory_mb', 0)
        
        if self.psutil_available and self.process is not None:
            return f"""
        ⏱️  **Duration:** {duration:.1f} seconds
        🧠  **Memory Usage:** {memory_increase:.1f} MB increase (Peak: {peak_memory:.1f} MB)
        """
        else:
            return f"""
        ⏱️  **Duration:** {duration:.1f} seconds
        🧠  **Memory Monitoring:** Not available (install psutil for memory tracking)
        """

def estimate_indexing_time(num_chunks: int, avg_chunk_length: int) -> Dict[str, float]:
    """Estimate indexing time based on document characteristics."""
    
    # Base estimates (in seconds)
    embedding_time_per_chunk = 0.1  # ~100ms per chunk for embedding generation
    pinecone_upsert_time_per_batch = 2.0  # ~2 seconds per batch of 100
    
    # Calculate estimates
    total_embedding_time = num_chunks * embedding_time_per_chunk
    num_batches = (num_chunks + 99) // 100  # Round up
    total_upsert_time = num_batches * pinecone_upsert_time_per_batch
    
    # Add overhead
    overhead_time = 10  # Index creation, model loading, etc.
    
    total_estimated_time = total_embedding_time + total_upsert_time + overhead_time
    
    return {
        'embedding_time': total_embedding_time,
        'upsert_time': total_upsert_time,
        'overhead_time': overhead_time,
        'total_time': total_estimated_time,
        'total_time_minutes': total_estimated_time / 60
    }
