
all: commit_msg hook

commit_msg:
	cd utils; sh apply_commit_message_template.sh

hook:
	cp utils/pre-commit.py .git/hooks/
