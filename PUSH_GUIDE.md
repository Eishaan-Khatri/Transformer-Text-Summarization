# Push Guide

The public repository currently exists but is empty:

```text
https://github.com/EishaanKhatri/Text_Summarization
```

After reviewing the local rebuild:

```powershell
cd D:\CV\portfolio\Text_Summarization
git init
git branch -M main
git add .
git commit -m "Rebuild transformer summarization benchmark"
git remote add origin https://github.com/EishaanKhatri/Text_Summarization.git
git push -u origin main
```

If the remote rejects because it already has commits, fetch first and inspect
before forcing anything.

