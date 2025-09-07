#!/usr/bin/env python3
"""
Distributed Execution System for LLM Risk Fairness Experiments

This module provides distributed experiment execution capabilities using Celery, Ray, and Dask
for scaling experiments across multiple workers and machines.
"""

import os
import json
import time
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import pickle

# Distributed computing imports
try:
    import celery
    from celery import Celery, group, chain, chord
    from celery.result import AsyncResult
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

try:
    import dask
    from dask.distributed import Client, as_completed, Future
    from dask import delayed
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

# Core experiment imports
from llm_risk_fairness_experiment import (
    ExperimentConfig, ExperimentProgress, LLMClient,
    make_stratified_subjects, run_llm_on_subject
)


@dataclass
class DistributedTaskConfig:
    """Configuration for distributed task execution."""
    backend: str = "celery"  # celery, ray, dask
    workers: int = 4
    batch_size: int = 10
    max_retries: int = 3
    retry_delay: float = 60.0
    timeout: float = 300.0
    heartbeat_interval: float = 30.0
    result_backend: str = "redis://localhost:6379/0"
    broker_url: str = "redis://localhost:6379/0"
    
    # Ray-specific settings
    ray_address: Optional[str] = None
    ray_num_cpus: Optional[int] = None
    ray_num_gpus: Optional[int] = None
    
    # Dask-specific settings
    dask_scheduler_address: Optional[str] = None
    dask_n_workers: Optional[int] = None


class DistributedExperimentManager:
    """Manager for distributed experiment execution."""
    
    def __init__(self, config: DistributedTaskConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize backend
        if config.backend == "celery" and CELERY_AVAILABLE:
            self._init_celery()
        elif config.backend == "ray" and RAY_AVAILABLE:
            self._init_ray()
        elif config.backend == "dask" and DASK_AVAILABLE:
            self._init_dask()
        else:
            raise ValueError(f"Backend {config.backend} not available or not supported")
    
    def _init_celery(self):
        """Initialize Celery backend."""
        self.celery_app = Celery(
            'llm_experiment_worker',
            broker=self.config.broker_url,
            backend=self.config.result_backend
        )
        
        # Configure Celery
        self.celery_app.conf.update(
            task_serializer='pickle',
            result_serializer='pickle',
            accept_content=['pickle'],
            result_expires=3600,
            task_routes={'*': {'queue': 'llm_experiments'}},
            worker_prefetch_multiplier=1,
            task_acks_late=True,
            worker_hijack_root_logger=False
        )
        
        # Define tasks
        @self.celery_app.task(bind=True, max_retries=self.config.max_retries)
        def process_subject_batch(self, batch_data: Dict) -> Dict:
            """Process a batch of subjects."""
            return self._process_batch(batch_data)
        
        self.process_subject_batch = process_subject_batch
    
    def _init_ray(self):
        """Initialize Ray backend."""
        ray_config = {}
        if self.config.ray_num_cpus:
            ray_config['num_cpus'] = self.config.ray_num_cpus
        if self.config.ray_num_gpus:
            ray_config['num_gpus'] = self.config.ray_num_gpus
        
        if self.config.ray_address:
            ray.init(address=self.config.ray_address, **ray_config)
        else:
            ray.init(**ray_config)
        
        # Define remote function
        @ray.remote
        def process_subject_batch_ray(batch_data: Dict) -> Dict:
            """Process a batch of subjects with Ray."""
            manager = DistributedExperimentManager.__new__(DistributedExperimentManager)
            return manager._process_batch(batch_data)
        
        self.process_subject_batch_ray = process_subject_batch_ray
    
    def _init_dask(self):
        """Initialize Dask backend."""
        if self.config.dask_scheduler_address:
            self.dask_client = Client(self.config.dask_scheduler_address)
        else:
            self.dask_client = Client(
                processes=True,
                n_workers=self.config.dask_n_workers or self.config.workers,
                threads_per_worker=2
            )
        
        # Define delayed function
        @delayed
        def process_subject_batch_dask(batch_data: Dict) -> Dict:
            """Process a batch of subjects with Dask."""
            manager = DistributedExperimentManager.__new__(DistributedExperimentManager)
            return manager._process_batch(batch_data)
        
        self.process_subject_batch_dask = process_subject_batch_dask
    
    def _process_batch(self, batch_data: Dict) -> Dict:
        """Process a batch of subjects (core processing logic)."""
        try:
            # Extract batch information
            subjects = batch_data['subjects']
            models = batch_data['models']
            experiment_config = ExperimentConfig(**batch_data['config'])
            batch_id = batch_data['batch_id']
            
            results = []
            batch_start_time = time.time()
            
            self.logger.info(f"Processing batch {batch_id} with {len(subjects)} subjects and {len(models)} models")
            
            # Process each subject-model combination
            for subject in subjects:
                for model in models:
                    try:
                        # Create LLM client
                        provider = "openai" if "gpt" in model else "anthropic" if "claude" in model else "google"
                        client = LLMClient(provider, model, experiment_config)
                        
                        # Run experiment
                        result = run_llm_on_subject(subject, client, experiment_config)
                        
                        # Add batch metadata
                        result['batch_id'] = batch_id
                        result['worker_id'] = os.getpid()
                        result['processing_time'] = time.time() - batch_start_time
                        
                        results.append(result)
                        
                        # Periodic heartbeat
                        if len(results) % 5 == 0:
                            self.logger.info(f"Batch {batch_id}: Processed {len(results)} subject-model combinations")
                    
                    except Exception as e:
                        self.logger.error(f"Error processing subject {subject.get('subject_id', 'unknown')} with model {model}: {str(e)}")
                        # Continue with other subjects
                        continue
            
            batch_time = time.time() - batch_start_time
            self.logger.info(f"Batch {batch_id} completed in {batch_time:.2f}s with {len(results)} results")
            
            return {
                'batch_id': batch_id,
                'results': results,
                'batch_time': batch_time,
                'success_count': len(results),
                'worker_id': os.getpid(),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {str(e)}")
            return {
                'batch_id': batch_data.get('batch_id', 'unknown'),
                'results': [],
                'error': str(e),
                'success_count': 0,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    def run_distributed_experiment(self, config: ExperimentConfig, outdir: str) -> Dict:
        """Run experiment using distributed execution."""
        start_time = datetime.now(timezone.utc)
        self.logger.info(f"Starting distributed experiment with {self.config.backend} backend")
        
        # Generate subjects
        if config.stratified_sampling:
            subjects = make_stratified_subjects(config.K)
        else:
            from llm_risk_fairness_experiment import make_subjects
            subjects = make_subjects(config.K)
        
        # Create batches
        batches = self._create_batches(subjects, config)
        total_tasks = len(batches) * len(config.models)
        
        self.logger.info(f"Created {len(batches)} batches for {total_tasks} total tasks")
        
        # Initialize progress tracking
        progress = ExperimentProgress(
            total_calls=total_tasks * config.repeats,
            completed_calls=0,
            failed_calls=0,
            start_time=start_time,
            last_checkpoint=start_time
        )
        
        # Submit tasks based on backend
        if self.config.backend == "celery":
            results = self._run_celery_experiment(batches, config, progress, outdir)
        elif self.config.backend == "ray":
            results = self._run_ray_experiment(batches, config, progress, outdir)
        elif self.config.backend == "dask":
            results = self._run_dask_experiment(batches, config, progress, outdir)
        else:
            raise ValueError(f"Unknown backend: {self.config.backend}")
        
        # Aggregate results
        return self._aggregate_results(results, config, outdir, start_time)
    
    def _create_batches(self, subjects: List[Dict], config: ExperimentConfig) -> List[Dict]:
        """Create batches of subjects for distributed processing."""
        batches = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(subjects), batch_size):
            batch_subjects = subjects[i:i + batch_size]
            batch_id = f"batch_{i//batch_size + 1:04d}"
            
            batch = {
                'batch_id': batch_id,
                'subjects': batch_subjects,
                'models': config.models,
                'config': asdict(config),
                'batch_index': i // batch_size,
                'total_batches': (len(subjects) + batch_size - 1) // batch_size
            }
            batches.append(batch)
        
        return batches
    
    def _run_celery_experiment(self, batches: List[Dict], config: ExperimentConfig, 
                              progress: ExperimentProgress, outdir: str) -> List[Dict]:
        """Run experiment using Celery."""
        self.logger.info("Submitting tasks to Celery workers")
        
        # Submit all batch tasks
        job_group = group(self.process_subject_batch.s(batch) for batch in batches)
        result = job_group.apply_async()
        
        # Monitor progress
        completed_results = []
        while not result.ready():
            completed_count = sum(1 for r in result.results if r.ready())
            progress.completed_calls = completed_count * config.repeats
            
            self.logger.info(f"Progress: {completed_count}/{len(batches)} batches completed")
            
            # Save checkpoint
            if completed_count > 0 and completed_count % 5 == 0:
                checkpoint_path = os.path.join(outdir, 'distributed_checkpoint.json')
                progress.save_checkpoint(checkpoint_path)
            
            time.sleep(self.config.heartbeat_interval)
        
        # Collect results
        for batch_result in result.results:
            try:
                completed_results.append(batch_result.get(timeout=self.config.timeout))
            except Exception as e:
                self.logger.error(f"Failed to get batch result: {str(e)}")
                completed_results.append({'results': [], 'error': str(e)})
        
        return completed_results
    
    def _run_ray_experiment(self, batches: List[Dict], config: ExperimentConfig,
                           progress: ExperimentProgress, outdir: str) -> List[Dict]:
        """Run experiment using Ray."""
        self.logger.info("Submitting tasks to Ray workers")
        
        # Submit all batch tasks
        futures = [self.process_subject_batch_ray.remote(batch) for batch in batches]
        
        # Monitor progress
        completed_results = []
        remaining_futures = futures[:]
        
        while remaining_futures:
            ready_futures, remaining_futures = ray.wait(
                remaining_futures, 
                num_returns=min(5, len(remaining_futures)),
                timeout=self.config.heartbeat_interval
            )
            
            # Collect ready results
            for future in ready_futures:
                try:
                    result = ray.get(future)
                    completed_results.append(result)
                except Exception as e:
                    self.logger.error(f"Failed to get Ray result: {str(e)}")
                    completed_results.append({'results': [], 'error': str(e)})
            
            # Update progress
            progress.completed_calls = len(completed_results) * config.repeats
            self.logger.info(f"Progress: {len(completed_results)}/{len(batches)} batches completed")
            
            # Save checkpoint
            if len(completed_results) > 0 and len(completed_results) % 5 == 0:
                checkpoint_path = os.path.join(outdir, 'distributed_checkpoint.json')
                progress.save_checkpoint(checkpoint_path)
        
        return completed_results
    
    def _run_dask_experiment(self, batches: List[Dict], config: ExperimentConfig,
                            progress: ExperimentProgress, outdir: str) -> List[Dict]:
        """Run experiment using Dask."""
        self.logger.info("Submitting tasks to Dask workers")
        
        # Submit all batch tasks
        futures = [self.process_subject_batch_dask(batch) for batch in batches]
        computed_futures = self.dask_client.compute(futures)
        
        # Monitor progress
        completed_results = []
        for future in as_completed(computed_futures):
            try:
                result = future.result()
                completed_results.append(result)
                
                # Update progress
                progress.completed_calls = len(completed_results) * config.repeats
                self.logger.info(f"Progress: {len(completed_results)}/{len(batches)} batches completed")
                
                # Save checkpoint
                if len(completed_results) % 5 == 0:
                    checkpoint_path = os.path.join(outdir, 'distributed_checkpoint.json')
                    progress.save_checkpoint(checkpoint_path)
                    
            except Exception as e:
                self.logger.error(f"Failed to get Dask result: {str(e)}")
                completed_results.append({'results': [], 'error': str(e)})
        
        return completed_results
    
    def _aggregate_results(self, batch_results: List[Dict], config: ExperimentConfig,
                          outdir: str, start_time: datetime) -> Dict:
        """Aggregate results from all batches."""
        all_results = []
        total_success = 0
        total_errors = 0
        batch_stats = []
        
        for batch_result in batch_results:
            if 'error' in batch_result:
                total_errors += 1
                self.logger.error(f"Batch {batch_result.get('batch_id', 'unknown')} failed: {batch_result['error']}")
            else:
                batch_results_data = batch_result.get('results', [])
                all_results.extend(batch_results_data)
                total_success += batch_result.get('success_count', 0)
                
                batch_stats.append({
                    'batch_id': batch_result.get('batch_id'),
                    'success_count': batch_result.get('success_count', 0),
                    'batch_time': batch_result.get('batch_time', 0),
                    'worker_id': batch_result.get('worker_id'),
                    'timestamp': batch_result.get('timestamp')
                })
        
        end_time = datetime.now(timezone.utc)
        total_time = (end_time - start_time).total_seconds()
        
        # Calculate statistics
        summary = {
            'experiment_name': f'distributed_experiment_{start_time.strftime("%Y%m%d_%H%M%S")}',
            'backend': self.config.backend,
            'total_batches': len(batch_results),
            'successful_batches': len(batch_results) - total_errors,
            'failed_batches': total_errors,
            'total_results': len(all_results),
            'total_time_seconds': total_time,
            'results_per_second': len(all_results) / total_time if total_time > 0 else 0,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'config': asdict(config),
            'distributed_config': asdict(self.config),
            'batch_statistics': batch_stats
        }
        
        # Save results
        os.makedirs(outdir, exist_ok=True)
        
        # Save raw results
        results_file = os.path.join(outdir, 'distributed_results.jsonl')
        with open(results_file, 'w') as f:
            for result in all_results:
                f.write(json.dumps(result) + '\n')
        
        # Save summary
        summary_file = os.path.join(outdir, 'distributed_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save batch statistics
        batch_stats_file = os.path.join(outdir, 'batch_statistics.json')
        with open(batch_stats_file, 'w') as f:
            json.dump(batch_stats, f, indent=2)
        
        self.logger.info(f"Distributed experiment completed: {len(all_results)} results in {total_time:.2f}s")
        self.logger.info(f"Results saved to {outdir}")
        
        return summary
    
    def shutdown(self):
        """Shutdown distributed backend."""
        if self.config.backend == "ray" and RAY_AVAILABLE:
            ray.shutdown()
        elif self.config.backend == "dask" and DASK_AVAILABLE:
            self.dask_client.close()
        # Celery workers are managed externally


class DistributedExperimentCLI:
    """Command-line interface for distributed experiments."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def start_workers(self, backend: str, num_workers: int = 4, **kwargs):
        """Start distributed workers."""
        if backend == "celery":
            self._start_celery_workers(num_workers, **kwargs)
        elif backend == "ray":
            self._start_ray_cluster(**kwargs)
        elif backend == "dask":
            self._start_dask_cluster(num_workers, **kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def _start_celery_workers(self, num_workers: int, broker_url: str = "redis://localhost:6379/0"):
        """Start Celery workers."""
        import subprocess
        import sys
        
        print(f"Starting {num_workers} Celery workers...")
        
        # Start workers
        processes = []
        for i in range(num_workers):
            cmd = [
                sys.executable, "-m", "celery", "worker",
                "-A", "distributed_execution",
                "-Q", "llm_experiments",
                "-n", f"worker{i}@%h",
                "--loglevel=info",
                f"--broker={broker_url}"
            ]
            
            process = subprocess.Popen(cmd)
            processes.append(process)
            print(f"Started worker {i} (PID: {process.pid})")
        
        try:
            for process in processes:
                process.wait()
        except KeyboardInterrupt:
            print("Stopping workers...")
            for process in processes:
                process.terminate()
    
    def _start_ray_cluster(self, address: Optional[str] = None, **kwargs):
        """Start Ray cluster."""
        if address:
            print(f"Connecting to Ray cluster at {address}")
            ray.init(address=address)
        else:
            print("Starting local Ray cluster")
            ray.init(**kwargs)
        
        print("Ray cluster started successfully")
        print(f"Ray dashboard: http://127.0.0.1:8265")
    
    def _start_dask_cluster(self, num_workers: int, **kwargs):
        """Start Dask cluster."""
        from dask.distributed import LocalCluster
        
        print(f"Starting Dask cluster with {num_workers} workers...")
        
        cluster = LocalCluster(
            n_workers=num_workers,
            processes=True,
            **kwargs
        )
        
        print(f"Dask cluster started at: {cluster.scheduler_address}")
        print(f"Dashboard: {cluster.dashboard_link}")
        
        # Keep cluster running
        try:
            input("Press Enter to shutdown cluster...\n")
        except KeyboardInterrupt:
            pass
        finally:
            cluster.close()


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Distributed LLM Risk Fairness Experiment")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Workers command
    workers_parser = subparsers.add_parser('workers', help='Start distributed workers')
    workers_parser.add_argument('--backend', choices=['celery', 'ray', 'dask'], default='celery')
    workers_parser.add_argument('--num-workers', type=int, default=4)
    workers_parser.add_argument('--broker-url', default='redis://localhost:6379/0')
    workers_parser.add_argument('--ray-address', help='Ray cluster address')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run distributed experiment')
    run_parser.add_argument('--config', required=True, help='Experiment config file')
    run_parser.add_argument('--outdir', required=True, help='Output directory')
    run_parser.add_argument('--backend', choices=['celery', 'ray', 'dask'], default='celery')
    run_parser.add_argument('--workers', type=int, default=4)
    run_parser.add_argument('--batch-size', type=int, default=10)
    run_parser.add_argument('--broker-url', default='redis://localhost:6379/0')
    run_parser.add_argument('--ray-address', help='Ray cluster address')
    run_parser.add_argument('--dask-address', help='Dask scheduler address')
    
    args = parser.parse_args()
    
    if args.command == 'workers':
        cli = DistributedExperimentCLI()
        cli.start_workers(
            backend=args.backend,
            num_workers=args.num_workers,
            broker_url=args.broker_url,
            address=args.ray_address
        )
    
    elif args.command == 'run':
        # Load experiment config
        from llm_risk_fairness_experiment import ExperimentConfig
        exp_config = ExperimentConfig.from_yaml(args.config)
        
        # Create distributed config
        dist_config = DistributedTaskConfig(
            backend=args.backend,
            workers=args.workers,
            batch_size=args.batch_size,
            broker_url=args.broker_url,
            ray_address=args.ray_address,
            dask_scheduler_address=args.dask_address
        )
        
        # Run distributed experiment
        manager = DistributedExperimentManager(dist_config)
        try:
            results = manager.run_distributed_experiment(exp_config, args.outdir)
            print(f"Experiment completed successfully!")
            print(f"Results: {results['total_results']} in {results['total_time_seconds']:.2f}s")
        finally:
            manager.shutdown()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()