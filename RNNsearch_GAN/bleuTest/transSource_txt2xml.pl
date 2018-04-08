#!/usr/bin/perl

($input, $output) = @ARGV;
open FIN,"<$input" or die $!;
open FOUT,">$output" or die $!;


print FOUT "<?xml version=\"1.0\" encoding=\"UTF-16\"?>\n";
print FOUT "<srcset setid=\"xxx\" srclang=\"Chinese\" trglang=\"English\">\n";
print FOUT "<doc docid=\"xxx\" sysid=\"ch\">\n";


$line_num = 1;
while($sline = <FIN>)
{
	chomp $sline;
	
	print FOUT "<seg id=$line_num>$sline<\/seg>\n";
	
}

print FOUT "<\/doc>\n<\/srcset>\n";
