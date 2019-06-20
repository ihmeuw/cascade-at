
dockerize: dismod_at.tgz
	docker build -t reg.ihme.washington.edu/dismod/dismod_at:19.05.21 .

dismod_at.tgz:
	-git clone git@github.com:bradbell/dismod_at.git
	git -C dismod_at pull
	tar zcf dismod_at.tgz dismod_at

clean:
	rm -f dismod_at.tgz
	rm -rf dismod_at
