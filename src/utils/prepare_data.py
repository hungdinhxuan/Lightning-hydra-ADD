def prepare_data(
    git_repo_url: str,
    dvc_data_path: str,
    pvc_path: str,
    git_branch: str = "main",
    type: str = "train"
    ) -> str:
    """
    Pull dataset from MinIO S3 using DVC and Git.
    
    Args:
        git_repo_url: Git repository URL (e.g., https://github.com/user/repo.git)
        dvc_data_path: Path to data folder in DVC (relative to repo root, e.g., 'data/eval')
        pvc_path: Path to PVC where data will be stored
        git_branch: Git branch to checkout (default: main)
    
    Returns:
        Path to the downloaded dataset
    """
    import os
    import subprocess
    import shutil
    from pathlib import Path
    
    # Create PVC directory if it doesn't exist
    os.makedirs(pvc_path, exist_ok=True)
    
    # Extract repo name from URL
    repo_name = git_repo_url.rstrip('/').split('/')[-1].replace('.git', '')
    repo_path = os.path.join(pvc_path, f"repo_{repo_name}")
    
    
    # Prepare authenticated Git URL for private repos
    auth_git_url = git_repo_url


    print(f"No git token provided, using environment variable GIT_TOKEN")
    git_token = os.getenv('GIT_TOKEN')
    if git_token:
        print(f"Git token: {git_token}")
        auth_git_url = git_repo_url.replace('https://', f'https://{git_token}@')
        print(f"Using authenticated access for private repository")
    else:
        print(f"No git token found in environment variables")
        raise ValueError("No git token found in environment variables")
    
    print(f"Cloning repository: {git_repo_url.split('@')[-1]}")  # Hide token in logs
    print(f"Target branch: {git_branch}")
    
    # Clone or update repository
    try:
        if os.path.exists(repo_path):
            # Check if it's a valid git repository
            git_dir = os.path.join(repo_path, '.git')
            if os.path.isdir(git_dir):
                print(f"Repository already exists at {repo_path}, pulling latest changes...")
                # Update remote URL with token if provided
                if git_token:
                    subprocess.run(
                        ['git', '-C', repo_path, 'remote', 'set-url', 'origin', auth_git_url],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                result = subprocess.run(
                    ['git', '-C', repo_path, 'pull', 'origin', git_branch],
                    check=True,
                    capture_output=True,
                    text=True
                )
                if result.stdout:
                    print(f"Git pull output: {result.stdout.strip()}")
            else:
                print(f"Directory exists but is not a valid git repository. Removing and cloning fresh...")
                shutil.rmtree(repo_path)
                print(f"Cloning repository...")
                result = subprocess.run(
                    ['git', 'clone', '-b', git_branch, auth_git_url, repo_path],
                    check=True,
                    capture_output=True,
                    text=True
                )
                if result.stderr:  # Git outputs to stderr even on success
                    print(f"Git clone output: {result.stderr.strip()}")
        else:
            print(f"Cloning repository...")
            result = subprocess.run(
                ['git', 'clone', '-b', git_branch, auth_git_url, repo_path],
                check=True,
                capture_output=True,
                text=True
            )
            if result.stderr:  # Git outputs to stderr even on success
                print(f"Git clone output: {result.stderr.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Git operation failed with exit code {e.returncode}")
        print(f"=== Stdout ===")
        print(e.stdout if e.stdout else "(empty)")
        print(f"=== Stderr ===")
        print(e.stderr if e.stderr else "(empty)")
        raise RuntimeError(
            f"Git clone/pull failed. Check: 1) Repository URL is correct, "
            f"2) Token has repo access (if private), 3) Branch '{git_branch}' exists"
        ) from e
    
    print(f"Repository cloned/updated successfully at {repo_path}")
    
    # Clean up token from git config for security
    if git_token:
        try:
            subprocess.run(
                ['git', '-C', repo_path, 'remote', 'set-url', 'origin', git_repo_url],
                check=True,
                capture_output=True,
                text=True
            )
            print("Git credentials cleaned up from repository config")
        except Exception as e:
            print(f"Warning: Could not clean up git credentials: {e}")
    
    # Configure DVC remote with MinIO S3
    # print(f"\n=== Configuring DVC Remote: {dvc_remote_name} ===")
    # print(f"S3 Endpoint: {s3_endpoint_url}")
    # print(f"S3 Bucket: s3://dvc-storage")
    
    # # Set up environment variables for DVC
    env = os.environ.copy()
   
    
    # Pull data from DVC
    print(f"\n=== Pulling Data from DVC ===")
    print(f"Data path: {dvc_data_path}")
    print(f"Running command: dvc pull {dvc_data_path}")
    
    # Check if DVC file exists in repo
        # Untar all tar.gz files in the data folder
    src_data_path = os.path.join(repo_path, dvc_data_path)
    print(f"Src data path: {src_data_path}")
    try:
        if os.path.exists(src_data_path):
            print(f"Src data path exists: {src_data_path}")
            for file in os.listdir(src_data_path):
                print(f"File: {file}")
                if file.endswith('.tar.gz') or file.endswith('.tar'):
                    folder_name = file.replace('.tar.gz', '').replace('.tar', '')
                    # Check if the folder already exists
                    if os.path.exists(os.path.join(src_data_path, folder_name)):
                        print(f"Folder {folder_name} already exists, skipping...")
                        return os.path.join(src_data_path, folder_name) if type == "train" else src_data_path
        else:
            result = subprocess.run(
                ['dvc', 'pull', dvc_data_path],
                cwd=repo_path,
                env=env,
                check=True,
                capture_output=True,
                text=True
            )
            
            # Log stdout if present
            if result.stdout:
                print("=== DVC Pull Output ===")
                print(result.stdout)
            
            # Log stderr if present (warnings, info messages)
            if result.stderr:
                print("=== DVC Pull Stderr ===")
                print(result.stderr)
            
            print(f"✓ Data pulled successfully from DVC")
            
    except subprocess.CalledProcessError as e:
        print(f"\n❌ DVC pull failed with exit code {e.returncode}")
        print(f"=== Command ===")
        print(f"dvc pull {dvc_data_path}")
        print(f"\n=== Working Directory ===")
        print(f"{repo_path}")
        print(f"\n=== Stdout ===")
        print(e.stdout if e.stdout else "(empty)")
        print(f"\n=== Stderr (Error Details) ===")
        print(e.stderr if e.stderr else "(empty)")
        print(f"\n=== Environment Variables ===")
        print(f"AWS_ACCESS_KEY_ID: {'***' if env.get('AWS_ACCESS_KEY_ID') else 'NOT SET'}")
        print(f"AWS_SECRET_ACCESS_KEY: {'***' if env.get('AWS_SECRET_ACCESS_KEY') else 'NOT SET'}")
        
        # Try to get more DVC info
        print(f"\n=== DVC Configuration ===")
        try:
            config_result = subprocess.run(
                ['dvc', 'remote', 'list'],
                cwd=repo_path,
                capture_output=True,
                text=True
            )
            print(config_result.stdout)
        except Exception as config_err:
            print(f"Could not get DVC config: {config_err}")
        
        # Re-raise with more context
        raise RuntimeError(
            f"DVC pull failed. See logs above for details. "
            f"Common causes: 1) Data not in remote storage, 2) Wrong credentials, "
            f"3) Wrong remote URL, 4) File path doesn't exist in DVC"
        ) from e
    

    for file in os.listdir(src_data_path):
        if file.endswith('.tar.gz') or file.endswith('.tar'):
            folder_name = file.replace('.tar.gz', '').replace('.tar', '')
            # Check if the folder already exists
            if os.path.exists(os.path.join(src_data_path, folder_name)):
                print(f"Folder {folder_name} already exists, skipping...")
                return os.path.join(src_data_path, folder_name) if type == "train" else src_data_path
            try:
                subprocess.run(
                    ['tar', '-xvf', os.path.join(src_data_path, file)],
                    check=True,
                    capture_output=True,
                    text=True,
                    cwd=src_data_path
                )
                print(f"Data untarred successfully from {os.path.join(src_data_path, file)}")
                return os.path.join(src_data_path, folder_name) if type == "train" else src_data_path
            except subprocess.CalledProcessError as e:
                print(f"❌ Tar operation failed with exit code {e.returncode}")
                print(f"=== Stdout ===")
                print(e.stdout if e.stdout else "(empty)")
                print(f"=== Stderr ===")
                print(e.stderr if e.stderr else "(empty)")
                raise RuntimeError(
                    f"Tar operation failed. See logs above for details. "
                    f"Common causes: 1) File not found, 2) Permission denied, 3) Invalid file format"
                ) from e
