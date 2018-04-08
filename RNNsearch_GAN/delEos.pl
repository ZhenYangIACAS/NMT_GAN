#!/usr/bin/perl
use strict;
(my $in, my $out)= @ARGV;
open IN, '<', $in or die $!;
open OUT, '>', $out or die $!;

while(<IN>){
	chomp;
	my @array = split /\s+/,$_;
	#shift @array;
	#pop @array;
	
	#print STDOUT shift @array;
	#print STDOUT "<EOS1>";
	
	my $num = 0;
	if(@array <= 2){
		print OUT "$_\n";
	}else{
		foreach my $word (@array){
			if($num == 0){
				if($word == '<EOS1>' && @array != 1){
				
				}else{
					print OUT "$word ";
				}
			}elsif($num == @array -1){
				if($word =='<EOS2>'){
					print OUT "\n";
				}else{
					print OUT "$word\n";
				}	
			}else{
				print OUT "$word ";
			}

			$num ++;
		}
	#print STDOUT " <EOS2>\n";
	}
}
