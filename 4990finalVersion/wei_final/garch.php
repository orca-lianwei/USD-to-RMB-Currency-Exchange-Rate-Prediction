<?php
$days = 7;

if(isset($_GET['predict_days'])){
	$days = $_GET['predict_days'];
}
$currency = 'CNY';
if(isset($_GET['currency'])){
    $currency = $_GET['currency'];
}
echo exec('python GARCH.py ' . $days . " " . $currency);
?>
