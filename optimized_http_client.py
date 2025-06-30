#!/usr/bin/env python3
"""
Optimized HTTP Client
Provides connection pooling, timeouts, retry logic, and async support for API calls.
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging
import time
from typing import Dict, Any, Optional, Union
from functools import wraps
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)

class OptimizedHTTPClient:
    """
    Optimized HTTP client with connection pooling, timeouts, and retry logic.
    """
    
    def __init__(self, 
                 max_retries: int = 3,
                 backoff_factor: float = 0.3,
                 timeout: tuple = (5, 30),  # (connect_timeout, read_timeout)
                 pool_connections: int = 10,
                 pool_maxsize: int = 20):
        
        self.timeout = timeout
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS"],
            backoff_factor=backoff_factor
        )
        
        # Configure adapters with connection pooling
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize
        )
        
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set default headers
        self.session.headers.update({
            'User-Agent': 'Football-Prediction-App/1.0',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        })
        
        logger.info(f"HTTP client initialized with {max_retries} retries, {timeout}s timeout")
    
    def get(self, url: str, params: Optional[Dict] = None, headers: Optional[Dict] = None, **kwargs) -> requests.Response:
        """Make GET request with optimizations."""
        try:
            start_time = time.time()
            
            # Merge headers
            request_headers = self.session.headers.copy()
            if headers:
                request_headers.update(headers)
            
            response = self.session.get(
                url, 
                params=params, 
                headers=request_headers,
                timeout=self.timeout,
                **kwargs
            )
            
            duration = time.time() - start_time
            logger.debug(f"GET {url} - {response.status_code} - {duration:.2f}s")
            
            response.raise_for_status()
            return response
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout for GET {url}")
            raise
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error for GET {url}")
            raise
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error for GET {url}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error for GET {url}: {e}")
            raise
    
    def post(self, url: str, data: Optional[Dict] = None, json: Optional[Dict] = None, 
             headers: Optional[Dict] = None, **kwargs) -> requests.Response:
        """Make POST request with optimizations."""
        try:
            start_time = time.time()
            
            # Merge headers
            request_headers = self.session.headers.copy()
            if headers:
                request_headers.update(headers)
            
            response = self.session.post(
                url, 
                data=data,
                json=json,
                headers=request_headers,
                timeout=self.timeout,
                **kwargs
            )
            
            duration = time.time() - start_time
            logger.debug(f"POST {url} - {response.status_code} - {duration:.2f}s")
            
            response.raise_for_status()
            return response
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout for POST {url}")
            raise
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error for POST {url}")
            raise
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error for POST {url}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error for POST {url}: {e}")
            raise
    
    def close(self):
        """Close the session and clean up resources."""
        self.session.close()
        logger.debug("HTTP client session closed")

class AsyncHTTPClient:
    """
    Async HTTP client for concurrent API calls.
    """
    
    def __init__(self, 
                 timeout: int = 30,
                 max_connections: int = 100,
                 max_connections_per_host: int = 10):
        
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.connector = aiohttp.TCPConnector(
            limit=max_connections,
            limit_per_host=max_connections_per_host,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        self.session = None
        logger.info(f"Async HTTP client initialized with {timeout}s timeout")
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=self.timeout,
            headers={
                'User-Agent': 'Football-Prediction-App/1.0',
                'Accept': 'application/json',
                'Accept-Encoding': 'gzip, deflate'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
        if self.connector:
            await self.connector.close()
    
    async def get(self, url: str, params: Optional[Dict] = None, headers: Optional[Dict] = None) -> Dict[str, Any]:
        """Make async GET request."""
        if not self.session:
            raise RuntimeError("Session not initialized. Use async with context manager.")
        
        try:
            start_time = time.time()
            
            async with self.session.get(url, params=params, headers=headers) as response:
                duration = time.time() - start_time
                logger.debug(f"Async GET {url} - {response.status} - {duration:.2f}s")
                
                response.raise_for_status()
                return await response.json()
                
        except asyncio.TimeoutError:
            logger.error(f"Timeout for async GET {url}")
            raise
        except aiohttp.ClientError as e:
            logger.error(f"Client error for async GET {url}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error for async GET {url}: {e}")
            raise
    
    async def post(self, url: str, data: Optional[Dict] = None, json: Optional[Dict] = None, 
                   headers: Optional[Dict] = None) -> Dict[str, Any]:
        """Make async POST request."""
        if not self.session:
            raise RuntimeError("Session not initialized. Use async with context manager.")
        
        try:
            start_time = time.time()
            
            async with self.session.post(url, data=data, json=json, headers=headers) as response:
                duration = time.time() - start_time
                logger.debug(f"Async POST {url} - {response.status} - {duration:.2f}s")
                
                response.raise_for_status()
                return await response.json()
                
        except asyncio.TimeoutError:
            logger.error(f"Timeout for async POST {url}")
            raise
        except aiohttp.ClientError as e:
            logger.error(f"Client error for async POST {url}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error for async POST {url}: {e}")
            raise

class APICallManager:
    """
    Manager for optimized API calls with rate limiting and concurrent execution.
    """
    
    def __init__(self, max_workers: int = 5, rate_limit_per_second: float = 2.0):
        self.http_client = OptimizedHTTPClient()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.rate_limit = rate_limit_per_second
        self.last_call_time = 0.0
        self._lock = threading.Lock()
        
        logger.info(f"API call manager initialized with {max_workers} workers, {rate_limit_per_second} calls/sec")
    
    def _apply_rate_limit(self):
        """Apply rate limiting to API calls."""
        with self._lock:
            current_time = time.time()
            time_since_last_call = current_time - self.last_call_time
            min_interval = 1.0 / self.rate_limit
            
            if time_since_last_call < min_interval:
                sleep_time = min_interval - time_since_last_call
                time.sleep(sleep_time)
            
            self.last_call_time = time.time()
    
    def make_api_call(self, url: str, params: Optional[Dict] = None, headers: Optional[Dict] = None) -> Dict[str, Any]:
        """Make a single API call with rate limiting."""
        self._apply_rate_limit()
        
        try:
            response = self.http_client.get(url, params=params, headers=headers)
            return response.json()
        except Exception as e:
            logger.error(f"API call failed for {url}: {e}")
            raise
    
    def make_concurrent_api_calls(self, api_calls: list) -> list:
        """Make multiple API calls concurrently."""
        """
        api_calls format: [
            {'url': 'http://...', 'params': {...}, 'headers': {...}},
            ...
        ]
        """
        futures = []
        
        for call in api_calls:
            future = self.executor.submit(
                self.make_api_call,
                call.get('url'),
                call.get('params'),
                call.get('headers')
            )
            futures.append(future)
        
        results = []
        for future in futures:
            try:
                result = future.result(timeout=60)  # 60 second timeout per call
                results.append(result)
            except Exception as e:
                logger.error(f"Concurrent API call failed: {e}")
                results.append(None)
        
        return results
    
    async def make_async_api_calls(self, api_calls: list) -> list:
        """Make multiple API calls asynchronously."""
        async with AsyncHTTPClient() as client:
            tasks = []
            
            for call in api_calls:
                task = asyncio.create_task(
                    client.get(
                        call.get('url'),
                        call.get('params'),
                        call.get('headers')
                    )
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and handle exceptions
            processed_results = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Async API call failed: {result}")
                    processed_results.append(None)
                else:
                    processed_results.append(result)
            
            return processed_results
    
    def close(self):
        """Clean up resources."""
        self.http_client.close()
        self.executor.shutdown(wait=True)
        logger.info("API call manager resources cleaned up")

# Decorators for easy integration

def with_timeout(timeout_seconds: int = 30):
    """Decorator to add timeout to any function."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {timeout_seconds} seconds")
            
            # Set timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            try:
                result = func(*args, **kwargs)
                signal.alarm(0)  # Cancel alarm
                return result
            except Exception as e:
                signal.alarm(0)  # Cancel alarm
                raise e
        
        return wrapper
    return decorator

def with_retry(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator to add retry logic to any function."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            current_delay = delay
            
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries >= max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries} retries: {e}")
                        raise e
                    
                    logger.warning(f"Function {func.__name__} failed (attempt {retries}/{max_retries}): {e}")
                    time.sleep(current_delay)
                    current_delay *= backoff
            
        return wrapper
    return decorator

# Global instances
http_client = OptimizedHTTPClient()
api_manager = APICallManager()

# Cleanup function for application shutdown
def cleanup_http_resources():
    """Clean up HTTP client resources."""
    http_client.close()
    api_manager.close()
    logger.info("HTTP resources cleaned up")