use warnings;
use strict;
use Data::Dumper;
use List::MoreUtils qw/uniq/;

my $dir = './';
opendir DIR, $dir or die $!;
my @files = readdir(DIR);
close DIR;

my %seen;
my @tags;
foreach my $file (@files){
	next unless $file =~ m/^gene2/g;
	my ($tag) = $file =~ m/gene2(\S+)_simple.txt/;
	push @tags, $tag;
	open FILE, $file or die $!;
	while(<FILE>){
		chomp;
		my($gene) = (split/\s+/,$_)[0];
		$seen{$tag}{$gene} = 1;
	}
	close FILE;
}

foreach my $tag (keys %seen){
    my $c = scalar keys $seen{$tag};
    print join("\t",$tag,$c),"\n";
}

open OUT, '>', 'common_genes.txt' or die $!;
my $tag1 = 'hpo';
foreach my $tag2 (@tags){
	next if $tag1 eq $tag2;
	foreach my $gene (keys $seen{$tag1}){
		print OUT join("\t",$tag1,$tag2,$gene),"\n" unless !defined $seen{$tag2}{$gene};	
	}		
}
close OUT;
