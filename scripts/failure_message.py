print(f"""
<!DOCTYPE html>
<html>
<body>
<h1 style="color:red">Build failed</h1>
<p>See changes: https://github.com/${{ github.repository }}/commit/${{github.sha}}</p>

</body>
</html>
""")
