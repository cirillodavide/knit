use warnings;
use strict;
use PerlIO::gzip;
use List::MoreUtils qw(uniq);
use Data::Dumper;

my %gene_to_go;
my %gene_to_hpo;

my $file = shift;

$/ = "";
my @para;
open my $in, '../go-basic.obo' or die $!;
while(<$in>){
    chomp;
    push @para, $_;
}
close $in;

my %domain;
foreach my $para(@para){
    my (@id, @namespace, @is_obsolete) = (0);
    $para =~ s/\n/\t/g;
    (@is_obsolete) = $para =~ /\s+is_obsolete:\s+(\S+)/g;
    next if grep{"true" eq $_}@is_obsolete; # remove obsolete terms
    (@id) = $para =~ /\s+id:\s+(\S+:\d+)\s+/g;
    (@namespace) = $para =~ /\s+namespace:\s+(\S+)\s+/g;
    $domain{"@id"} = "@namespace";
}

$/ = "\n";
my $file_go = '../goa_human.gaf.gz';
open my $in_go, '<:gzip', $file_go or die $!;
while(<$in_go>){
    chomp;
    next if $_ =~ /^\!/;
    my ($geneName) = (split/\t/,$_)[2];
        my ($go) = $_ =~ /(GO:\d+)/g; # ignoring colocalizes_with
        my ($ec) = (split/\t/,$_)[6];
        next if !grep{$ec eq $_} qw/EXP IDA IPI IMP IGI IEP/; # only experimental evidence codes
        #next if !grep{$ec eq $_} qw/IDA/;
        push @{$gene_to_go{$geneName}}, $go;
        @{$gene_to_go{$geneName}} = uniq @{$gene_to_go{$geneName}};
    }
close $in_go;

open GO_OUT, '>', 'gene2goBP_simple.txt' or die $!;
foreach my $gene (keys %gene_to_go){
    foreach my $term (@{$gene_to_go{$gene}}){
        print GO_OUT join("\t", $gene,$term),"\n" if $domain{$term} eq "biological_process"; 
    }
}
close GO_OUT;

open GO_OUT, '>', 'gene2goMF_simple.txt' or die $!;
foreach my $gene (keys %gene_to_go){
    foreach my $term (@{$gene_to_go{$gene}}){
        print GO_OUT join("\t", $gene,$term),"\n" if $domain{$term} eq "molecular_function"; 
    }
}
close GO_OUT;

open GO_OUT, '>', 'gene2goCC_simple.txt' or die $!;
foreach my $gene (keys %gene_to_go){
    foreach my $term (@{$gene_to_go{$gene}}){
        print GO_OUT join("\t", $gene,$term),"\n" if $domain{$term} eq "cellular_component"; 
    }
}
close GO_OUT;


my $file_hpo = '../OMIM_ALL_FREQUENCIES_diseases_to_genes_to_phenotypes.txt';
open my $in_hpo, $file_hpo or die $!;
while(<$in_hpo>){
    chomp;
    next if $. < 2;
    my ($geneName, $hp) = (split/\t/,$_)[1,3];
    push @{$gene_to_hpo{$geneName}}, $hp;
    @{$gene_to_hpo{$geneName}} = uniq @{$gene_to_hpo{$geneName}};
}
close $in_hpo;

open HPO_OUT, '>', 'gene2hpo_simple.txt' or die $!;
foreach my $gene (keys %gene_to_hpo){
    foreach my $term (@{$gene_to_hpo{$gene}}){
        print HPO_OUT join("\t", $gene,$term),"\n";
    }
}
close HPO_OUT;

