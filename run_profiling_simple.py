#!/usr/bin/env python3
"""
Performance Profiling Runner (Simple Version)
This script runs the training environment profiling and saves results to files
without any special characters that could cause encoding issues.
"""

import subprocess
import sys
import os
import argparse
from datetime import datetime
import json

def run_profiling(symbol='GBPUSD', algorithm='ppo', output_dir='profiling_results'):
    """
    Run performance profiling and save results to files
    
    Args:
        symbol (str): Trading symbol to profile
        algorithm (str): Algorithm to profile (ppo or dqn)
        output_dir (str): Directory to save profiling results
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Starting performance profiling for {symbol} with {algorithm.upper()}")
    print(f"Results will be saved to: {output_dir}")
    print(f"Timestamp: {timestamp}")
    print("-" * 60)
    
    try:
        # Run the profiling command
        cmd = [
            sys.executable, 'train_from_csv.py',
            '--symbol', symbol,
            '--algorithm', algorithm,
            '--profile'
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        print("-" * 60)
        
        # Run profiling and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=300  # 5 minute timeout
        )
        
        # Save stdout to text file
        stdout_file = os.path.join(output_dir, f'profile_stdout_{symbol}_{algorithm}_{timestamp}.txt')
        with open(stdout_file, 'w', encoding='utf-8') as f:
            f.write(f"Profiling Results for {symbol} {algorithm.upper()}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write("=" * 80 + "\n\n")
            f.write(result.stdout)
        
        # Save stderr to separate file if there are errors
        if result.stderr:
            stderr_file = os.path.join(output_dir, f'profile_stderr_{symbol}_{algorithm}_{timestamp}.txt')
            with open(stderr_file, 'w', encoding='utf-8') as f:
                f.write(f"Error Output for {symbol} {algorithm.upper()}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write("=" * 80 + "\n\n")
                f.write(result.stderr)
            print(f"WARNING: Errors captured in: {stderr_file}")
        
        # Save summary to JSON for easy analysis
        summary = {
            'symbol': symbol,
            'algorithm': algorithm,
            'timestamp': timestamp,
            'return_code': result.returncode,
            'stdout_file': stdout_file,
            'stderr_file': stderr_file if result.stderr else None,
            'command': ' '.join(cmd),
            'success': result.returncode == 0
        }
        
        summary_file = os.path.join(output_dir, f'profile_summary_{symbol}_{algorithm}_{timestamp}.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Print results summary
        print("\n" + "=" * 60)
        print("PROFILING COMPLETED")
        print("=" * 60)
        print(f"Return code: {result.returncode}")
        print(f"Output saved to: {stdout_file}")
        if result.stderr:
            print(f"Errors saved to: {stderr_file}")
        print(f"Summary saved to: {summary_file}")
        
        # Extract and display key performance metrics
        if result.stdout:
            print("\nKEY PERFORMANCE METRICS:")
            print("-" * 40)
            
            lines = result.stdout.split('\n')
            for line in lines:
                if any(keyword in line for keyword in [
                    'Average time per step',
                    'TOP PERFORMANCE BOTTLENECKS',
                    'MEMORY USAGE',
                    'Peak:',
                    'Average:'
                ]):
                    print(line.strip())
        
        return True, stdout_file
        
    except subprocess.TimeoutExpired:
        print("ERROR: Profiling timed out after 5 minutes")
        return False, None
    except Exception as e:
        print(f"ERROR: Error running profiling: {e}")
        return False, None

def analyze_profiling_results(output_file):
    """
    Analyze profiling results and provide insights
    
    Args:
        output_file (str): Path to profiling output file
    """
    if not os.path.exists(output_file):
        print(f"ERROR: Output file not found: {output_file}")
        return
    
    print(f"\nANALYZING RESULTS: {output_file}")
    print("=" * 60)
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract key metrics
        lines = content.split('\n')
        
        # Find performance bottlenecks
        bottlenecks = []
        in_bottlenecks = False
        
        for line in lines:
            if 'TOP PERFORMANCE BOTTLENECKS' in line:
                in_bottlenecks = True
                continue
            elif in_bottlenecks and line.strip() and not line.startswith('  '):
                in_bottlenecks = False
                continue
            
            if in_bottlenecks and line.strip().startswith('  '):
                bottlenecks.append(line.strip())
        
        if bottlenecks:
            print("PERFORMANCE BOTTLENECKS:")
            for bottleneck in bottlenecks[:5]:  # Top 5
                print(f"  {bottleneck}")
        
        # Find memory usage
        memory_lines = [line for line in lines if 'MEMORY USAGE' in line or 'Peak:' in line or 'Average:' in line]
        if memory_lines:
            print("\nMEMORY USAGE:")
            for line in memory_lines:
                print(f"  {line.strip()}")
        
        # Find timing information
        timing_lines = [line for line in lines if 'ms' in line and ('avg' in line or 'total' in line)]
        if timing_lines:
            print("\nTIMING BREAKDOWN:")
            for line in timing_lines[:10]:  # Top 10 timing lines
                print(f"  {line.strip()}")
        
        # Provide recommendations
        print("\nOPTIMIZATION RECOMMENDATIONS:")
        print("-" * 40)
        
        if any('_get_technical_indicators' in line for line in bottlenecks):
            print("* Technical indicator calculations are the main bottleneck")
            print("* Consider implementing caching or vectorization")
            print("* Look into Numba JIT compilation for speedup")
        
        if any('_assemble_state' in line for line in bottlenecks):
            print("* State assembly is taking significant time")
            print("* Optimize data structure operations")
            print("* Consider pre-computing static parts of state")
        
        if any('ms avg' in line for line in timing_lines):
            avg_times = []
            for line in timing_lines:
                if 'ms avg' in line:
                    try:
                        time_str = line.split('ms avg')[0].split()[-1]
                        avg_times.append(float(time_str))
                    except:
                        continue
            
            if avg_times:
                max_time = max(avg_times)
                if max_time > 100:
                    print(f"* Maximum average time: {max_time:.2f}ms - needs optimization")
                elif max_time > 50:
                    print(f"* Maximum average time: {max_time:.2f}ms - moderate optimization needed")
                else:
                    print(f"* Maximum average time: {max_time:.2f}ms - performance looks good")
        
    except Exception as e:
        print(f"ERROR: Error analyzing results: {e}")

def main():
    """Main function to run profiling with command line arguments"""
    parser = argparse.ArgumentParser(description='Run performance profiling for trading environment')
    parser.add_argument('--symbol', type=str, default='GBPUSD', help='Trading symbol (default: GBPUSD)')
    parser.add_argument('--algorithm', type=str, default='ppo', choices=['ppo', 'dqn'], help='Algorithm to profile (default: ppo)')
    parser.add_argument('--output-dir', type=str, default='profiling_results', help='Output directory (default: profiling_results)')
    parser.add_argument('--analyze', type=str, help='Analyze existing profiling results file')
    
    args = parser.parse_args()
    
    if args.analyze:
        # Analyze existing results
        analyze_profiling_results(args.analyze)
    else:
        # Run new profiling
        success, output_file = run_profiling(args.symbol, args.algorithm, args.output_dir)
        
        if success and output_file:
            print(f"\nProfiling completed successfully!")
            print(f"Results saved to: {output_file}")
            
            # Ask if user wants to analyze results
            try:
                response = input("\nWould you like to analyze the results? (y/n): ").lower().strip()
                if response in ['y', 'yes']:
                    analyze_profiling_results(output_file)
            except KeyboardInterrupt:
                print("\n\nProfiling completed. Check the output files for results.")

if __name__ == "__main__":
    main() 