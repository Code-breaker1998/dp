<?php
header("Access-Control-Allow-Origin: *");
$site=$_POST['url'];
// echo $site;
$html = file_get_contents($site);
//echo $html;
$bytes=file_put_contents('markup.txt', $html);

// Can use this if your default interpreter is Python 2.x.
// Has some problem executing 'which python2'. So, absolute path is just simpler.
//$python_path=exec("which python 2>&1 ");
//$decision=exec("$python_path test.py $site 2>&1 ");

// Replace the path with the path of your python2.x installation.
//$decision=exec("C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Python 3.8\Python test.py $site 2>&1 ");
$decision=shell_exec("python test.py $site 2>&1");
echo $decision;
?>
