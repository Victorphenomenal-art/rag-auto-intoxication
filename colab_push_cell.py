# ============================================================
# COLAB CELL: clone this repo's contents into Colab, then push
# your own copy to a NEW GitHub repo you create first.
# ============================================================
#
# SECURITY NOTE: the version of this cell you were given elsewhere
# hardcoded a GitHub token as a plain string:
#     GITHUB_TOKEN = "github_pat_xxxxxxxxxx"
# Do NOT do this. Colab notebooks get shared, screenshotted, and
# sometimes accidentally committed themselves -- a hardcoded token in a
# cell is a token that WILL leak eventually. Use one of the two options
# below instead. Either way: create a FINE-GRAINED token scoped to only
# this one repo, with only "Contents: read/write" permission, and an
# expiry date -- not a classic token with broad access.
#
# Option A (recommended): Colab's built-in Secrets manager
#   1. Click the key icon in the left sidebar of Colab.
#   2. Add a secret named GITHUB_TOKEN, paste your token as the value.
#   3. Toggle "Notebook access" on for this notebook.
#   4. The code below reads it via `userdata.get(...)` -- never printed,
#      never saved into the notebook file itself.
#
# Option B: paste it interactively via getpass (not saved anywhere,
#   but you'll need to re-paste it each session).

import os
import subprocess

try:
    from google.colab import userdata
    GITHUB_TOKEN = userdata.get("GITHUB_TOKEN")  # Option A
except Exception:
    from getpass import getpass
    GITHUB_TOKEN = getpass("GitHub token (fine-grained, repo-scoped): ")  # Option B

GITHUB_USERNAME = input("GitHub username: ").strip()
REPO_NAME = input("New repo name (must already exist on GitHub, empty): ").strip()
GITHUB_EMAIL = input("Git commit email: ").strip()
GITHUB_NAME = input("Git commit name: ").strip()

assert GITHUB_TOKEN, "No token provided -- aborting."
assert GITHUB_USERNAME and REPO_NAME, "Username/repo name required -- aborting."

# --- Assumes you've already `!git clone`'d or uploaded this repo's files
#     into the current directory (e.g. via `!unzip rag_auto_intoxication.zip`
#     or by cloning the repo you're distributing this from). This cell only
#     handles committing + pushing, not scaffolding the files themselves. ---

os.system("git init")
os.system(f'git config user.email "{GITHUB_EMAIL}"')
os.system(f'git config user.name "{GITHUB_NAME}"')
os.system("git add .")
os.system('git commit -m "Initial research scaffold: exact ODEs, eviction policies, scale benchmarks"')
os.system("git branch -M main")

remote_url = f"https://{GITHUB_USERNAME}:{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git"
os.system(f"git remote add origin {remote_url}")

# Push, then immediately scrub the token out of git's remote config so it
# doesn't linger in .git/config in plaintext for the rest of the session.
push_result = subprocess.run(["git", "push", "-u", "origin", "main"], capture_output=True, text=True)
print(push_result.stdout, push_result.stderr)
os.system(f"git remote set-url origin https://github.com/{GITHUB_USERNAME}/{REPO_NAME}.git")

print(f"\nDone (check output above for errors). Repo: "
      f"https://github.com/{GITHUB_USERNAME}/{REPO_NAME}")
print("Token has been scrubbed from git remote config for this session, "
      "but rotate/revoke it after use if you're on a shared Colab instance.")
