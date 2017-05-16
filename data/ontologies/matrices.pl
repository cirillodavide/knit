use warnings;
use strict;
use Data::Dumper;

my $dir = 'res/res_simple';
opendir DIR, $dir or die $!;
my @files = readdir(DIR);
close DIR;

my %common_genes;
open FILE, $dir."/common_genes.txt" or die $!;
while(<FILE>){
	chomp;
	my($tag1,$tag2,$gene) = split/\s+/,$_;
	push @{$common_genes{$tag1}{$tag2}}, $gene;
}
close FILE;

foreach my $file (@files){
	next unless $file =~ m/^gene2/g;
	next if $file =~ m/^gene2hpo/g;
	my ($tag) = $file =~ m/gene2(\S+)_simple.txt/;
	
	my %genes;
	foreach my $gene (@{$common_genes{'hpo'}{$tag}}){
		$genes{$gene} = 1;
	}
	
	print $tag."-hpo\n";
	my $ref_mat = commonTerms($dir."/".$file, \%genes);
	my %mat = %$ref_mat;
	open MAT, '>', $tag."-hpo_mat.txt" or die $!;
	foreach my $geneA (keys %mat){
		foreach my $geneB (keys $mat{$geneA}){
			my $rate = $mat{$geneA}{$geneB};
			#if($geneA eq $geneB){$rate = 0;}
        	print MAT join("\t",$geneA,$geneB,$rate),"\n";
		}
	}
	close MAT;

	print "hpo-".$tag."\n";
	$ref_mat = commonTerms($dir."/gene2hpo_simple.txt", \%genes);
	%mat = %$ref_mat;
	open MAT, '>', "hpo-".$tag."_mat.txt" or die $!;
	foreach my $geneA (keys %mat){
		foreach my $geneB (keys $mat{$geneA}){
			my $rate = $mat{$geneA}{$geneB};
			#if($geneA eq $geneB){$rate = 0;}
        	print MAT join("\t",$geneA,$geneB,$rate),"\n";
		}
	}
	close MAT;
}


sub commonTerms {

	my ($file, $ref_genes) = @_;
	my %genes = %$ref_genes;

	my %HoA;
	open FILE, $file or die $!;
	while(<FILE>){
		chomp;
		my($gene,$term)=(split/\s+/,$_);
		next unless defined $genes{$gene};
		push @{$HoA{$gene}}, $term;
	}
	close FILE;

	my %mat;
	foreach my $geneA (keys %HoA){
		foreach my $geneB (keys %HoA){
			#next if $genaA eq $geneB; # exclude autogenous pairs
			my %in_array2 = map { $_ => 1 } @{$HoA{$geneB}};
			my @array3 = grep { $in_array2{$_} } @{$HoA{$geneA}};
			my %seen;
			my @unique = grep{!$seen{$_}++} @array3;
			my $comm = scalar @unique;
 			next if $comm == 0; # exclude pairs with no common terms (we don't want tons of zeros...)
			$mat{$geneA}{$geneB} = $comm;
		}
	}
	
	return \%mat;	
}
