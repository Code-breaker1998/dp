<?php
header("Access-Control-Allow-Origin: *");
$site=$_POST['url'];
// echo $site;
$html = file_get_contents($site);
echo $site;
?>
