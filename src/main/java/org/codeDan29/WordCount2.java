package org.codeDan29;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.util.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Counter;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.partition.HashPartitioner;
import org.apache.hadoop.mapreduce.lib.partition.KeyFieldBasedComparator;
import org.apache.hadoop.mapreduce.lib.partition.KeyFieldBasedPartitioner;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.util.StringUtils;

public class WordCount2 {
    public static class TokenizerMapper extends Mapper<Object, Text, Text, IntWritable> {
        static enum CountersEnum { INPUT_WORDS }
        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();
        private boolean caseSensitive;
        private Set<String> patternsToSkip = new HashSet<String>();
        private Configuration conf;
        private BufferedReader fis;

        @Override
        public void setup(Context context) throws IOException, InterruptedException {
            conf = context.getConfiguration();
            caseSensitive = conf.getBoolean("wordcount.case.sensitive", true);
            if (conf.getBoolean("wordcount.skip.patterns", false)) {
                URI[] patternsURIs = Job.getInstance(conf).getCacheFiles();
                for (URI patternsURI : patternsURIs) {
                    Path patternsPath = new Path(patternsURI.getPath());
                    String patternsFileName = patternsPath.getName().toString();
                    parseSkipFile(patternsFileName);
                }
            }
        }

        private void parseSkipFile(String fileName) {
            try {
                fis = new BufferedReader(new FileReader(fileName));
                String pattern = null;
                while ((pattern = fis.readLine()) != null) {
                    patternsToSkip.add(pattern);
                }
            } catch (IOException ioe) {
                System.err.println("Caught exception while parsing the cached file '"
                        + StringUtils.stringifyException(ioe));
            } finally {
                try {
                    if (fis != null) {
                        fis.close();
                    }
                } catch (IOException e) {
                    System.err.println("Caught exception while closing file '"
                            + StringUtils.stringifyException(e));
                }
            }
        }

        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String line = (caseSensitive) ? value.toString() : value.toString().toLowerCase();

            StringBuilder modifiedLine = new StringBuilder(line);
            for (String pattern : patternsToSkip) {
                modifiedLine = modifiedLine.replace(0, modifiedLine.length(), modifiedLine.toString().replaceAll(pattern, ""));
            }
            line = modifiedLine.toString();

            StringTokenizer itr = new StringTokenizer(line);
            while (itr.hasMoreTokens()) {
                String token = itr.nextToken().toLowerCase();
                word.set(token);
                context.write(word, one);
                Counter counter = context.getCounter(CountersEnum.class.getName(), CountersEnum.INPUT_WORDS.toString());
                counter.increment(1);
            }
        }
    }

    public static class IntSumReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        private IntWritable result = new IntWritable();

        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, result);
        }
    }

    public static class DescendingKeyPartitioner extends HashPartitioner<Text, IntWritable> {
        @Override
        public int getPartition(Text key, IntWritable value, int numReduceTasks) {
            // Ensure that keys with the same first letter go to the same reducer
            return super.getPartition(new Text(key.toString().substring(0, 1)), value, numReduceTasks);
        }
    }

    public static class DescendingKeyComparator extends KeyFieldBasedComparator {
        @Override
        public int compare(byte[] b1, int s1, int l1, byte[] b2, int s2, int l2) {
            // Reverse the comparison
            return -super.compare(b1, s1, l1, b2, s2, l2);
        }
    }

    public static class Pair<L extends Comparable<L>, R extends Comparable<R>> {
        private L left;
        private R right;

        public Pair() {
        }

        public Pair(L left, R right) {
            this.left = left;
            this.right = right;
        }

        public L getLeft() {
            return left;
        }

        public void setLeft(L left) {
            this.left = left;
        }

        public R getRight() {
            return right;
        }

        public void setRight(R right) {
            this.right = right;
        }
    }

    public static class TopWordsReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        private PriorityQueue<Pair<String, Integer>> topWordsQueue;

        @Override
        public void setup(Context context) {
            // Initialize the priority queue with a custom comparator
            topWordsQueue = new PriorityQueue<>(Comparator.comparingInt(Pair::getRight));
        }

        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }

            topWordsQueue.offer(new Pair<>(key.toString(), sum));

            while (topWordsQueue.size() > 50) {
                topWordsQueue.poll();
            }
        }

        @Override
        public void cleanup(Context context) throws IOException, InterruptedException {
            // Emit the final top 50 results
            Stack<Pair<String, Integer>> stack = new Stack<>();

            while (!topWordsQueue.isEmpty()) {
                stack.push(topWordsQueue.poll());
            }

            while (!stack.isEmpty()) {
                Pair<String, Integer> pair = stack.pop();
                context.write(new Text(pair.getLeft()), new IntWritable(pair.getRight()));
            }
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        GenericOptionsParser optionParser = new GenericOptionsParser(conf, args);
        String[] remainingArgs = optionParser.getRemainingArgs();
        if ((remainingArgs.length != 2) && (remainingArgs.length != 4)) {
            System.err.println("Usage: wordcount <in> <out> [-skip skipPatternFile]");
            System.exit(2);
        }

        Job job = Job.getInstance(conf, "word count");
        job.setJarByClass(WordCount2.class);
        job.setMapperClass(TokenizerMapper.class);
        job.setCombinerClass(IntSumReducer.class);
        job.setReducerClass(TopWordsReducer.class); // Change this line
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        List<String> otherArgs = new ArrayList<String>();
        for (int i = 0; i < remainingArgs.length; ++i) {
            if ("-skip".equals(remainingArgs[i])) {
                job.addCacheFile(new Path(remainingArgs[++i]).toUri());
                job.getConfiguration().setBoolean("wordcount.skip.patterns", true);
            } else {
                otherArgs.add(remainingArgs[i]);
            }
        }

        FileInputFormat.addInputPath(job, new Path(otherArgs.get(0)));
        FileOutputFormat.setOutputPath(job, new Path(otherArgs.get(1)));

        // Set up secondary sort
        job.setPartitionerClass(DescendingKeyPartitioner.class);
        job.setSortComparatorClass(DescendingKeyComparator.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);

        KeyFieldBasedPartitioner.PARTITIONER_OPTIONS = "-k1,1";
        KeyFieldBasedComparator.COMPARATOR_OPTIONS = "-k1,1";

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}