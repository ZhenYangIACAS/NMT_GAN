#!/usr/bin/perl
($input, $output) = @ARGV;
open IN, '<', $input or die $!;
open OUT, '>', $output or die $!;

while(<IN>)
{
	chomp();
	
	if(/<seg id=/)
	{
		$_ =~ s/<seg id=\"(\w+)\">/<seg id=$1>/;
		print OUT "$_\n"; 
	}
	else
	{
		print OUT "$_\n";
	}
	
}
